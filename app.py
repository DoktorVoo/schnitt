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
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Weißer Hintergrund wird entfernt
    # Alles was hell (Papier) ist (>240), wird schwarz. Der Rest (Karten) wird weiß.
    # Dadurch trennen sich Karten mit weißem Abstand dazwischen automatisch auf.
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Konturen finden
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Konturen sortieren (Grobe y-Zeilenbildung, dann x-Achse)
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1] // 200, cv2.boundingRect(c)[0]))

    cards = []
    card_count = 0
    base_name = os.path.splitext(os.path.basename(filename))[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Rauschen und winzige Fragmente ignorieren
        if w < 100 or h < 100:
            continue

        extracted_region = img[y:y + h, x:x + w]

        # 2. Mathematischer Grid-Slicer (Raster-Erkennung)
        # Wir ermitteln das beste Raster (Spalten x Zeilen) für diesen Kasten, 
        # basierend auf dem typischen Magic-Karten-Verhältnis (Höhe ist ca. 1.4x die Breite).
        best_cols = 1
        best_rows = 1
        best_error = 9999
        
        for cols in range(1, 5):
            for rows in range(1, 5):
                card_w = w / cols
                card_h = h / rows
                ratio = card_h / card_w
                
                # Wie weit weicht dieses Raster vom perfekten MTG-Kartenformat ab?
                error = abs(ratio - 1.4)
                
                if error < best_error:
                    best_error = error
                    best_cols = cols
                    best_rows = rows
                    
        # 3. Den gefundenen Kasten zerschneiden
        step_x = w // best_cols
        step_y = h // best_rows
        
        for r in range(best_rows):
            for c in range(best_cols):
                card_img = extracted_region[r*step_y:(r+1)*step_y, c*step_x:(c+1)*step_x]
                
                card_count += 1
                name, data = save_to_buffer(card_img, f"{base_name}_{card_count}.jpg")
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
