import os
import zipfile
import io

import numpy as np
import cv2
from flask import Flask, request, send_file
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024  # 250 MB

def save_to_buffer(cv2_img, name):
    # Konvertierung von BGR (OpenCV) zu RGB (PIL) für den JPEG-Export
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, 'JPEG', quality=95, optimize=True)
    buf.seek(0)
    return name, buf.read()

def process_image(image_bytes, filename):
    # Bild aus dem Speicher laden
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    # Vorbereitung für die Konturenerkennung
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Leichtes Weichzeichnen, um Rauschen zu reduzieren
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Schwellenwert: Alles was dunkel ist (schwarze Rahmen) wird weiß markiert
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    # Konturen finden
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Konturen von oben nach unten, dann links nach rechts sortieren
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    cards = []
    card_count = 0
    base_name = os.path.splitext(os.path.basename(filename))[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter: Ignoriere zu kleine Fragmente (Dreck/Ränder)
        if w < 150 or h < 150:
            continue

        # Bild der gefundenen Box ausschneiden
        extracted_region = img[y:y + h, x:x + w]

        # Logik für Split-Karten: 
        # Wenn die Breite deutlich größer ist als die Höhe, ist es ein Kartenpaar (Split)
        if w > h * 1.2:
            # In der Mitte teilen
            mid = w // 2
            left_part = extracted_region[:, 0:mid]
            right_part = extracted_region[:, mid:w]

            for part in (left_part, right_part):
                card_count += 1
                name, data = save_to_buffer(part, f"{base_name}_{card_count}.jpg")
                cards.append((name, data))
        else:
            # Normale hochkant-Karte
            card_count += 1
            name, data = save_to_buffer(extracted_region, f"{base_name}_{card_count}.jpg")
            cards.append((name, data))

    return cards

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "Kein File hochgeladen", 400

        output_buffer = io.BytesIO()

        with zipfile.ZipFile(file, 'r') as in_zip:
            with zipfile.ZipFile(output_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as out_zip:
                for name in in_zip.namelist():
                    # Nur Bilder verarbeiten, __MACOSX Ordner ignorieren
                    if name.lower().endswith(('.jpg', '.jpeg', '.png')) and '__MACOSX' not in name:
                        image_data = in_zip.read(name)
                        card_list = process_image(image_data, name)
                        for c_name, c_data in card_list:
                            out_zip.writestr(c_name, c_data)

        output_buffer.seek(0)
        return send_file(
            output_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='karten_geschnitten.zip'
        )

    return '''
    <!doctype html>
    <title>MTG Card Splitter</title>
    <style>
        body { font-family: sans-serif; text-align: center; padding-top: 50px; background: #f4f4f4; }
        .box { background: white; padding: 30px; border-radius: 8px; display: inline-block; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        input[type="file"] { margin-bottom: 20px; }
    </style>
    <div class="box">
        <h1>MTG Karten-Cutter</h1>
        <p>Lade ein ZIP mit deinen Scans hoch.</p>
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept=".zip" /><br>
          <input type="submit" value="ZIP verarbeiten & Download" style="padding: 10px 20px; cursor: pointer;">
        </form>
    </div>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
