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
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, 'JPEG', quality=95, optimize=True)
    buf.seek(0)
    return name, buf.read()


def trim_white_borders(image):
    """
    Schneidet reinweiße Ränder von einem Bild ab.
    Sucht nach den äußersten nicht-weißen Pixeln (wie z.B. dem schwarzen Kartenrand).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Alles was heller als 240 (fast reines Weiß) ist, wird schwarz. Alles andere weiß.
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Koordinaten aller nicht-weißen Pixel auf der Maske finden
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Wenn das resultierende Bild Sinn macht, zuschneiden
        if w > 50 and h > 50:
            return image[y:y+h, x:x+w]
    return image


def process_image(image_bytes, filename):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Konturen sortieren (Zeilenweise, dann Spaltenweise)
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1] // 200, cv2.boundingRect(c)[0]))

    cards = []
    card_count = 0
    base_name = os.path.splitext(os.path.basename(filename))[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 100 or h < 100:
            continue

        extracted_region = img[y:y + h, x:x + w]

        # Raster-Erkennung (Grid-Slicer)
        best_cols = 1
        best_rows = 1
        best_error = 9999
        
        for cols in range(1, 5):
            for rows in range(1, 5):
                card_w = w / cols
                card_h = h / rows
                ratio = card_h / card_w
                error = abs(ratio - 1.4) # MTG Karten haben ein ~1:1.4 Verhältnis
                
                if error < best_error:
                    best_error = error
                    best_cols = cols
                    best_rows = rows
                    
        step_x = w // best_cols
        step_y = h // best_rows
        
        for r in range(best_rows):
            for c in range(best_cols):
                card_img = extracted_region[r*step_y:(r+1)*step_y, c*step_x:(c+1)*step_x]
                
                # --- NEU: Den weißen Rand entfernen, falls nach dem Raster-Schnitt noch was übrig ist ---
                card_img = trim_white_borders(card_img)
                
                card_count += 1
                name, data = save_to_buffer(card_img, f"{base_name}_{card_count}.jpg")
                cards.append((name, data))

    return cards


def create_3x3_sheets(image_tuples):
    """
    Nimmt eine Liste von (Dateiname, Bild-Bytes), normiert deren Größe 
    und fügt sie in 3x3 DIN A4-freundliche Druckbögen zusammen.
    """
    # Alphabetisch sortieren, damit Serien zusammenbleiben
    image_tuples.sort(key=lambda x: x[0])
    
    # Standardgröße für eine MTG Karte (Hohe Auflösung, exakt 1 : 1.4 Verhältnis)
    CARD_W, CARD_H = 750, 1050 
    SHEET_W = CARD_W * 3
    SHEET_H = CARD_H * 3
    
    sheets = []
    current_sheet = Image.new('RGB', (SHEET_W, SHEET_H), (255, 255, 255))
    card_idx = 0
    sheet_idx = 1
    
    for name, data in image_tuples:
        try:
            # Bild laden und auf die exakte Kartengröße ziehen
            img = Image.open(io.BytesIO(data)).convert('RGB')
            # Image.Resampling.LANCZOS ist die korrekte Methode für Pillow 10+ [cite: 1]
            img = img.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS) 
            
            # Position im 3x3 Raster berechnen
            row = card_idx // 3
            col = card_idx % 3
            
            # Karte auf den Bogen kleben
            current_sheet.paste(img, (col * CARD_W, row * CARD_H))
            card_idx += 1
            
            # Wenn 9 Karten voll sind -> Bogen speichern
            if card_idx == 9:
                buf = io.BytesIO()
                current_sheet.save(buf, 'JPEG', quality=95)
                sheets.append((f"Druckbogen_{sheet_idx:03d}.jpg", buf.getvalue()))
                sheet_idx += 1
                card_idx = 0
                current_sheet = Image.new('RGB', (SHEET_W, SHEET_H), (255, 255, 255))
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {name}: {e}")
            
    # Den letzten, eventuell nicht ganz vollen Bogen speichern
    if card_idx > 0:
        buf = io.BytesIO()
        current_sheet.save(buf, 'JPEG', quality=95)
        sheets.append((f"Druckbogen_{sheet_idx:03d}.jpg", buf.getvalue()))
        
    return sheets


@app.route('/split', methods=['POST'])
def split_cards():
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
        download_name='karten_einzeln.zip'
    )


@app.route('/merge', methods=['POST'])
def merge_cards():
    file = request.files.get('file')
    if not file:
        return "Kein File hochgeladen", 400

    output_buffer = io.BytesIO()
    uploaded_images = []

    # ZIP einlesen
    with zipfile.ZipFile(file, 'r') as in_zip:
        for name in in_zip.namelist():
            if name.lower().endswith(('.jpg', '.jpeg', '.png')) and '__MACOSX' not in name:
                uploaded_images.append((name, in_zip.read(name)))

    # Bilder zu Druckbögen zusammenfügen
    sheets = create_3x3_sheets(uploaded_images)

    # Neues ZIP mit den Druckbögen erstellen
    with zipfile.ZipFile(output_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as out_zip:
        for s_name, s_data in sheets:
            out_zip.writestr(s_name, s_data)

    output_buffer.seek(0)
    return send_file(
        output_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='karten_druckboegen_3x3.zip'
    )


@app.route('/', methods=['GET'])
def index():
    return '''
    <!doctype html>
    <title>MTG Proxy Tool</title>
    <style>
        body { font-family: sans-serif; text-align: center; padding-top: 50px; background: #2c3e50; color: #ecf0f1; }
        .container { display: flex; justify-content: center; gap: 40px; flex-wrap: wrap; }
        .box { background: #34495e; padding: 30px; border-radius: 8px; width: 350px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
        h1 { margin-bottom: 40px; }
        h2 { color: #3498db; }
        input[type="file"] { margin: 20px 0; width: 100%; }
        input[type="submit"] { background: #3498db; color: white; border: none; padding: 10px 20px; font-weight: bold; border-radius: 4px; cursor: pointer; width: 100%; transition: 0.2s;}
        input[type="submit"]:hover { background: #2980b9; }
    </style>
    
    <h1>MTG Proxy Werkstatt</h1>
    
    <div class="container">
        <div class="box">
            <h2>1. Scans zerschneiden</h2>
            <p>Lädt ein ZIP mit eingescannten Seiten hoch und zerschneidet sie in Einzelkarten (ohne weißen Rand).</p>
            <form action="/split" method="post" enctype="multipart/form-data">
              <input type="file" name="file" accept=".zip" required />
              <input type="submit" value="In Einzelkarten zerlegen">
            </form>
        </div>

        <div class="box">
            <h2>2. Zu 3x3 zusammenfügen</h2>
            <p>Lädt ein ZIP mit Einzelkarten hoch und generiert druckfertige 3x3 Raster (ideal für DIN A4).</p>
            <form action="/merge" method="post" enctype="multipart/form-data">
              <input type="file" name="file" accept=".zip" required />
              <input type="submit" value="Druckbögen generieren">
            </form>
        </div>
    </div>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
