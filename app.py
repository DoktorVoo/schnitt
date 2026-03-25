import os
import zipfile
import io
import gc

import numpy as np
import cv2
from flask import Flask, request, send_file
from PIL import Image

app = Flask(__name__)
# Limit auf 250MB Upload, aber wir verarbeiten vorsichtiger
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024

def save_to_buffer(cv2_img, name):
    # Konvertierung zu PIL für sauberes JPEG
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, 'JPEG', quality=92, optimize=True)
    buf.seek(0)
    return name, buf.read()

def smart_trim(image):
    """
    Entfernt weiße Ränder. Schneidet ca. 2% des Randes weg,
    um Scan-Artefakte sicher zu eliminieren.
    """
    if image is None or image.size == 0: return image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 215 ist eine gute Schwelle für "fast weißes" Papier
    _, mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
    
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # Etwas großzügigerer Schnitt nach innen (2% Rasur)
        shave_w = int(w * 0.02)
        shave_h = int(h * 0.02)
        
        new_x, new_y = x + shave_w, y + shave_h
        new_w, new_h = w - (2 * shave_w), h - (2 * shave_h)
        
        if new_w > 50 and new_h > 50:
            return image[new_y:new_y+new_h, new_x:new_x+new_w]
    return image

def process_image(image_bytes, filename):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return []

    # SPEICHER-OPTIMIERUNG: 
    # Wenn das Bild extrem groß ist (z.B. > 3500px), skalieren wir es runter,
    # um den RAM-Tod auf Render.com zu verhindern.
    h_orig, w_orig = img.shape[:2]
    if w_orig > 3000:
        scale = 3000 / w_orig
        img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    card_count = 0
    base_name = os.path.splitext(os.path.basename(filename))[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 150 or h < 150: continue

        block = img[y:y + h, x:x + w]
        current_ratio = h / w
        
        # Erkennung: Einzelkarte oder Grid?
        # MTG Kartenratio ist ~1.4
        if 1.3 < current_ratio < 1.5:
            best_cols, best_rows = 1, 1
        else:
            best_cols, best_rows = 1, 1
            best_error = abs(current_ratio - 1.4)
            for cols in range(1, 5):
                for rows in range(1, 5):
                    if cols == 1 and rows == 1: continue
                    ratio = (h / rows) / (w / cols)
                    error = abs(ratio - 1.4)
                    if error < best_error * 0.85:
                        best_error = error
                        best_cols, best_rows = cols, rows

        # Karten ausschneiden
        sw, sh = w // best_cols, h // best_rows
        for r in range(best_rows):
            for c in range(best_cols):
                card_img = block[r*sh:(r+1)*sh, c*sw:(c+1)*sw]
                card_img = smart_trim(card_img)
                
                card_count += 1
                name, data = save_to_buffer(card_img, f"{base_name}_{card_count}.jpg")
                cards.append((name, data))
                
                # RAM aufräumen
                del card_img
        
        # RAM aufräumen nach jedem Block
        del block
        gc.collect()

    del img, gray, thresh
    gc.collect()
    return cards

def create_3x3_sheets(image_tuples):
    image_tuples.sort(key=lambda x: x[0])
    # Standard Magic-Größe für 300dpi Druck
    CARD_W, CARD_H = 750, 1050 
    SHEET_W, SHEET_H = CARD_W * 3, CARD_H * 3
    
    sheets = []
    current_sheet = Image.new('RGB', (SHEET_W, SHEET_H), (255, 255, 255))
    card_idx, sheet_idx = 0, 1
    
    for name, data in image_tuples:
        try:
            img = Image.open(io.BytesIO(data)).convert('RGB')
            img = img.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)
            
            x_pos = (card_idx % 3) * CARD_W
            y_pos = (card_idx // 3) * CARD_H
            current_sheet.paste(img, (x_pos, y_pos))
            
            card_idx += 1
            if card_idx == 9:
                buf = io.BytesIO()
                current_sheet.save(buf, 'JPEG', quality=90)
                sheets.append((f"Druckbogen_{sheet_idx:03d}.jpg", buf.getvalue()))
                sheet_idx += 1; card_idx = 0
                current_sheet = Image.new('RGB', (SHEET_W, SHEET_H), (255, 255, 255))
            
            del img
            gc.collect()
        except: continue
            
    if card_idx > 0:
        buf = io.BytesIO()
        current_sheet.save(buf, 'JPEG', quality=90)
        sheets.append((f"Druckbogen_{sheet_idx:03d}.jpg", buf.getvalue()))
    
    return sheets

@app.route('/split', methods=['POST'])
def split_cards():
    file = request.files.get('file')
    if not file: return "Kein File", 400
    
    output_buffer = io.BytesIO()
    with zipfile.ZipFile(file, 'r') as in_zip:
        with zipfile.ZipFile(output_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as out_zip:
            for name in in_zip.namelist():
                if name.lower().endswith(('.jpg', '.jpeg', '.png')) and '__MACOSX' not in name:
                    # Jedes Bild einzeln verarbeiten, um RAM zu sparen
                    image_data = in_zip.read(name)
                    results = process_image(image_data, name)
                    for c_name, c_data in results:
                        out_zip.writestr(c_name, c_data)
                    # Nach jedem Hauptbild aufräumen
                    del image_data, results
                    gc.collect()

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
        for s_name, s_data in sheets:
            out_zip.writestr(s_name, s_data)
    
    output_buffer.seek(0)
    return send_file(output_buffer, mimetype='application/zip', as_attachment=True, download_name='druckboegen.zip')

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>MTG Tool v3.2 (Safe Mode)</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #121212; color: #e0e0e0; padding-top: 50px; }
        .box { background: #1e1e1e; padding: 25px; border-radius: 12px; display: inline-block; width: 320px; margin: 15px; border: 1px solid #333; }
        h3 { color: #4dabf7; }
        input[type="submit"] { background: #1971c2; color: white; border: none; padding: 12px; cursor: pointer; width: 100%; border-radius: 6px; font-weight: bold; }
        input[type="submit"]:hover { background: #1864ab; }
        p { font-size: 0.85em; color: #aaa; }
    </style>
    <h1>MTG Proxy Tool v3.2</h1>
    <div class="box">
        <h3>1. Scans splitten</h3>
        <p>Zerschneidet Seiten & entfernt Ränder aggressiv. (Optimiert für wenig RAM)</p>
        <form action="/split" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".zip"><br><br>
            <input type="submit" value="Split & Trim">
        </form>
    </div>
    <div class="box">
        <h3>2. Zu 3x3 mergen</h3>
        <p>Erstellt Druckbögen aus Einzelkarten.</p>
        <form action="/merge" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".zip"><br><br>
            <input type="submit" value="Druckbögen erstellen">
        </form>
    </div>
    '''

if __name__ == "__main__":
    # Port 10000 ist Standard für Render
    app.run(host="0.0.0.0", port=10000)
