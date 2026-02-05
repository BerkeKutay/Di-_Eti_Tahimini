from flask import Flask, render_template, request
from ultralytics import YOLO
import os, uuid, cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("models/best.pt")
print("MODEL CLASSES:", model.names)

CLASS_COLORS = {
    "saglikli": (0, 200, 0),
    "hafif_gingivitis": (0, 165, 255),
    "ileri_gingivitis": (0, 0, 255),
    "kanama": (0, 0, 180),
    "periodontitis": (128, 0, 128),
    "plak": (255, 255, 0),
    "tartar": (255, 0, 0),
}

INFO = {
    "saglikli": (
        "Diş etleri sağlıklı görünüyor.",
        "Günde 2 kez fırçalama ve diş ipi yeterlidir.",
        False
    ),
    "hafif_gingivitis": (
        "Hafif düzeyde diş eti iltihabı başlangıcı tespit edildi.",
        "Ağız hijyenini artırın, 2–3 hafta izleyin.",
        False
    ),
    "ileri_gingivitis": (
        "İleri seviye diş eti iltihabı tespit edildi.",
        "Profesyonel diş hekimi kontrolü gerekir.",
        True
    ),
    "periodontitis": (
        "Kemik kaybı riski olan ciddi diş eti hastalığı tespit edildi.",
        "Acil diş hekimi müdahalesi gereklidir.",
        True
    ),
    "plak": (
        "Diş yüzeyinde plak birikimi tespit edildi.",
        "Düzenli fırçalama ve diş ipi önerilir.",
        False
    ),
    "tartar": (
        "Diş taşı oluşumu gözlemlendi.",
        "Diş taşı temizliği için diş hekimine gidilmelidir.",
        True
    ),
}

def dis_numarasi_tahmin(cene, yon, bolge):
    if cene == "Üst çene":
        return "11–13" if yon == "Sağ" else "21–23"
    else:
        return "41–43" if yon == "Sağ" else "31–33"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        filename = f"{uuid.uuid4().hex}.jpg"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        image = cv2.imread(upload_path)
        h, w = image.shape[:2]

        results = model(upload_path)[0]
        overlay = image.copy()

        unique = {}
        toplam_risk = 0

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.50:
                continue

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id].replace(" ", "_")

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1+x2)/2, (y1+y2)/2

            cene = "Üst çene" if cy < h/2 else "Alt çene"
            yon = "Sol" if cx < w/2 else "Sağ"
            bolge = "Ön dişler" if w*0.3 < cx < w*0.7 else "Arka dişler"
            konum = f"{cene} – {yon} {bolge}"
            dis_no = dis_numarasi_tahmin(cene, yon, bolge)

            risk = "Yüksek Güven" if conf >= 0.85 else "Orta Güven"

            aciklama, oneriler, uyari = INFO.get(
                class_name,
                ("Açıklama yok.", "Bir uzmana danışın.", True)
            )

            aciklama = f"{konum} bölgesinde {aciklama.lower()}"

            if class_name not in unique or conf > unique[class_name]["conf"]:
                unique[class_name] = {
                    "name": class_name,
                    "conf": round(conf*100, 1),
                    "konum": konum,
                    "dis_no": dis_no,
                    "risk": risk,
                    "aciklama": aciklama,
                    "oneriler": oneriler,
                    "uyari": uyari
                }

            color = CLASS_COLORS.get(class_name, (0,255,0))
            cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
            cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
            cv2.rectangle(image, (x1,y1), (x2,y2), color, 4)

            toplam_risk += conf

        detections = list(unique.values())
        detections.sort(key=lambda x: x["conf"], reverse=True)

        genel_saglik = max(0, 100 - int((toplam_risk/len(detections))*100)) if detections else 100

        result_name = f"result_{filename}"
        cv2.imwrite(os.path.join(RESULT_FOLDER, result_name), image)

        return render_template(
            "result.html",
            image=result_name,
            detections=detections,
            skor=genel_saglik
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
