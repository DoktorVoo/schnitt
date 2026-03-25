import os
import zipfile
import io

import numpy as np
import cv2
from flask import Flask, request, send_file
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024

def save_to_buffer(cv2_img, name):
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, 'JPEG', quality=95, optimize=True)
    buf.seek(0)
    return name, buf.read()

def smart_trim(image):
    """
    Findet den schwarzen Rahmen der Karte und schneidet den weißen Rest weg.
    Zusätzlich wird der Rand um 1.5% 'rasiert', um Restweiß zu killen.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Erhöhte Toleranz für 'nicht ganz weißes' Papier
    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Sicherheits-Marge: 1.5% nach innen rücken (Rasur)
        margin_w = int(w * 0.015)
        margin_h = int(h * 0.015)
        
        # Zuschneiden mit Sicherheitsabstand nach innen
        new_y = max(0, y + margin_h)
        new_h = max(1, h - (2 * margin_h))
        new_x = max(0, x + margin_w)
        new_w = max(1, w - (2 * margin_w))
        
        return image[new_y:new_y+new_h, new_x:new_x+new_w]
    return image

def process_image(image_bytes, filename):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return []

    # Wir suchen nach den Kartenblöcken auf der Seite
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    card_count = 0
    base_name = os.path.splitext(os.path.basename(filename))[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 200 or h < 200: continue

        # Extrahiere den Block (kann 1 Karte oder 9 Karten sein)
        block = img[y:y + h, x:x + w]
        
        # Entscheiden: Ist das schon EINE Karte oder ein Gitter?
        # Eine MTG Karte hat ein Verhältnis von ca. 1.39 bis 1.41
        current_ratio = h / w
        
        # Wenn das Verhältnis schon fast 1.4 ist, NICHT weiter aufteilen!
        if 1.3 < current_ratio < 1.5:
            best_cols, best_rows = 1, 1
        else:
            # Ansonsten Raster suchen
            best_cols, best_rows = 1, 1
            best_error = abs(current_ratio - 1.4)
            
            for cols in range(1, 5):
                for rows in range(1, 5):
                    if cols == 1 and rows == 1: continue
                    ratio = (h / rows) / (w / cols)
                    error = abs(ratio - 1.4)
                    # Kleiner Bonus für 1x1 oder bekannte Formate, um Fehl-Splits zu vermeiden
                    if error < best_error * 0.8: 
                        best_error = error
                        best_cols, best_rows = cols
                        best_rows = rows

        step_x, step_y = w // best_cols, h // best_rows
        for r in range(best_rows):
            for c in range(best_cols):
                card_img = block[r*step_y:(r+1)*step_y, c*step_x:(c+1)*step_x]
                # Jetzt das aggressive Trimming anwenden
                card_img = smart_trim(card_img)
                
                card_count += 1
                name, data = save_to_buffer(card_img, f"{base_name}_{card_count}.jpg")
                cards.append((name, data))

    return cards

def create_3x3_sheets(image_tuples):
    image_tuples.sort(key=lambda x: x[0])
    CARD_W, CARD_H = 750, 1050 
    SHEET_W, SHEET_H = CARD_W * 3, CARD_H * 3
    
    sheets = []
    current_sheet = Image.new('RGB', (SHEET_W, SHEET_H), (255, 255, 255))
    card_idx, sheet_idx = 0, 1
    
    for name, data in image_tuples:
        try:
            img = Image.open(io.BytesIO(data)).convert('RGB')
            img = img.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)
            current_sheet.paste(img, ((card_idx % 3) * CARD_W, (card_idx // 3) * CARD_H))
            card_idx += 1
            if card_idx == 9:
                buf = io.BytesIO()
                current_sheet.save(buf, 'JPEG', quality=95)
                sheets.append((f"Bogen_{sheet_idx:03d}.jpg", buf.getvalue()))
                sheet_idx += 1; card_idx = 0
                current_sheet = Image.new('RGB', (SHEET_W, SHEET_H), (255, 255, 255))
        except: continue
            
    if card_idx > 0:
        buf = io.BytesIO()
        current_sheet.save(buf, 'JPEG', quality=95)
        sheets.append((f"Bogen_{sheet_idx:03d}.jpg", buf.getvalue()))
    return sheets

@app.route('/split', methods=['POST'])
def split_cards():
    file = request.files.get('file'); 
    if not file: return "Kein File", 400
    output_buffer = io.BytesIO()
    with zipfile.ZipFile(file, 'r') as in_zip:
        with zipfile.ZipFile(output_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as out_zip:
            for name in in_zip.namelist():
                if name.lower().endswith(('.jpg', '.jpeg', '.png')) and '__MACOSX' not in name:
                    for c_name, c_data in process_image(in_zip.read(name), name):
                        out_zip.writestr(c_name, c_data)
    output_buffer.seek(0)
    return send_file(output_buffer, mimetype='application/zip', as_attachment=True, download_name='einzelkarten.zip')

@app.route('/merge', methods=['POST'])
def merge_cards():
    file = request.files.get('file')
    if not file: return "Kein File", 400
    imgs = []
    with zipfile.ZipFile(file, 'r') as in_zip:
        for name in in_zip.namelist():
            if name.lower().endswith(('.jpg', '.jpeg', '.png')) and '__MACOSX' not in name:
                imgs.append((name, in_zip.read(name)))
    sheets = create_3x3_sheets(imgs)
    output_buffer = io.BytesIO()
    with zipfile.ZipFile(output_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as out_zip:
        for s_name, s_data in sheets: out_zip.writestr(s_name, s_data)
    output_buffer.seek(0)
    return send_file(output_buffer, mimetype='application/zip', as_attachment=True, download_name='druckboegen.zip')

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>MTG Proxy Tool v3</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #1a1a1a; color: white; padding-top: 50px; }
        .box { background: #333; padding: 20px; border-radius: 10px; display: inline-block; width: 300px; margin: 10px; vertical-align: top; }
        input[type="submit"] { background: #007bff; color: white; border: none; padding: 10px; cursor: pointer; width: 100%; border-radius: 5px; }
    </style>
    <h1>MTG Proxy Tool v3</h1>
    <div class="box">
        <h3>1. Scans splitten</h3>
        <p><small>Zerschneidet Seiten & entfernt Ränder</small></p>
        <form action="/split" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".zip"><br><br>
            <input type="submit" value="Splitten & Trimmen">
        </form>
    </div>
    <div class="box">
        <h3>2. Zu 3x3 mergen</h3>
        <p><small>Erstellt fertige Druckbögen</small></p>
        <form action="/merge" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".zip"><br><br>
            <input type="submit" value="Druckbögen erstellen">
        </form>
    </div>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
