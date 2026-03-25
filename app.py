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
    # OpenCV -> PIL für JPEG-Export
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, 'JPEG', quality=90, optimize=True)
    buf.seek(0)
    return name, buf.read()


def process_image(image_bytes, filename):
    # Bild laden
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Schwarze Rahmen finden
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Konturen der Karten
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cards = []
    card_count = 0
    base_name = os.path.splitext(os.path.basename(filename))[0]

    # Nur diese Dateien behandeln wir als „zwei Karten nebeneinander“
    is_split_page = base_name.startswith("Split_1") or base_name.startswith("Split_2")

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Rauschen ignorieren
        if w < 100 or h < 100:
            continue

        card_img = img[y:y + h, x:x + w]

        if is_split_page:
            # Jede gefundene Karte in zwei Hälften teilen
            mid = w // 2
            left_part = card_img[:, 0:mid]
            right_part = card_img[:, mid:w]
            for part in (left_part, right_part):
                card_count += 1
                name, data = save_to_buffer(
                    part, f"{base_name}_split_{card_count}.jpg"
                )
                cards.append((name, data))
        else:
            # Alle anderen Seiten: genau eine Karte pro Kontur
            card_count += 1
            name, data = save_to_buffer(
                card_img, f"{base_name}_{card_count}.jpg"
            )
            cards.append((name, data))

    return cards


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "Kein File", 400

        output_buffer = io.BytesIO()

        # Eingeladenes ZIP lesen und neues ZIP mit zugeschnittenen Karten schreiben
        with zipfile.ZipFile(file, 'r') as in_zip:
            with zipfile.ZipFile(
                output_buffer, 'w', compression=zipfile.ZIP_DEFLATED
            ) as out_zip:
                for name in in_zip.namelist():
                    if (
                        name.lower().endswith(('.jpg', '.jpeg', '.png'))
                        and '__MACOSX' not in name
                    ):
                        card_list = process_image(in_zip.read(name), name)
                        for c_name, c_data in card_list:
                            out_zip.writestr(c_name, c_data)

        output_buffer.seek(0)
        return send_file(
            output_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='karten_geschnitten.zip'
        )

    # Einfaches Upload-Formular
    return '''
    <html>
      <body>
        <h1>MTG Karten zuschneiden</h1>
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="file" />
          <input type="submit" value="Upload ZIP" />
        </form>
      </body>
    </html>
    '''


if __name__ == "__main__":
    # Für lokales Testen
    app.run(host="0.0.0.0", port=5000, debug=True)