import os
import zipfile
import io
import gc

import numpy as np
import cv2
from flask import Flask, request, send_file
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024

def smart_trim(image):
    """ Entfernt weiße Ränder mit 3% Sicherheits-Schnitt nach innen. """
    if image is None or image.size == 0: return image
    h_img, w_img = image.shape[:2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aggressiverer Schwellenwert gegen graue Scans (200 statt 225)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # 3% Rasur für absolute Sauberkeit
        shave_w, shave_h = int(w * 0.03), int(h * 0.03)
        
        nx, ny = max(0, x + shave_w), max(0, y + shave_h)
        nw, nh = max(1, w - (2 * shave_w)), max(1, h - (2 * shave_h))
        return image[ny:ny+nh, nx:nx+nw]
    return image

def process_image(image_bytes, filename):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return []

    # RAM-Schutz: Skalieren falls riesig
    h_orig, w_orig = img.shape[:2]
    if w_orig > 2500:
        scale = 2500 / w_orig
        img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY_INV)
    
    # NEU: Morphologische Operation, um Karten-Inhalte zu "verschmelzen"
    # Verhindert, dass Planeswalker in 9 Teile zerfallen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cards = []
    base_name = os.path.splitext(os.path.basename(filename))[0]
    count = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 150 or h < 150: continue

        block = img[y:y + h, x:x + w]
        ratio = h / w
        
        # Grid-Erkennung (nur wenn es kein Einzelkarten-Format ist)
        if 1.25 < ratio < 1.55:
            cols, rows = 1, 1
        else:
            cols, rows = 1, 1
            best_err = abs(ratio - 1.4)
            for c in range(1, 4):
                for r in range(1, 4):
                    if c == 1 and r == 1: continue
                    err = abs(((h/r)/(w/c)) - 1.4)
                    if err < best_err * 0.9:
                        best_err, cols, rows = err, c, r

        sw, sh = w // cols, h // rows
        for r_idx in range(rows):
            for c_idx in range(cols):
                card = block[r_idx*sh:(r_idx+1)*sh, c_idx*sw:(c_idx+1)*sw]
                card = smart_trim(card)
                
                count += 1
                pil_card = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                pil_card.save(buf, 'JPEG', quality=85)
                cards.append((f"{base_name}_{count}.jpg", buf.getvalue()))
                
        gc.collect()

    return cards

@app.route('/split', methods=['POST'])
def split_cards():
    file = request.files.get('file')
    if not file: return "Upload fehlt", 400
    
    out_buf = io.BytesIO()
    with zipfile.ZipFile(file, 'r') as in_zip:
        with zipfile.ZipFile(out_buf, 'w', compression=zipfile.ZIP_DEFLATED) as out_zip:
            for name in in_zip.namelist():
                if name.lower().endswith(('.jpg', '.jpeg', '.png')) and '__MACOSX' not in name:
                    results = process_image(in_zip.read(name), name)
                    for c_name, c_data in results:
                        out_zip.writestr(c_name, c_data)
                    gc.collect()

    out_buf.seek(0)
    return send_file(out_buf, mimetype='application/zip', as_attachment=True, download_name='karten.zip')

@app.route('/merge', methods=['POST'])
def merge_cards():
    file = request.files.get('file')
    if not file: return "Upload fehlt", 400
    
    imgs = []
    with zipfile.ZipFile(file, 'r') as in_zip:
        for name in in_zip.namelist():
            if name.lower().endswith(('.jpg', '.jpeg', '.png')) and '__MACOSX' not in name:
                imgs.append((name, in_zip.read(name)))

    imgs.sort(key=lambda x: x[0])
    CW, CH = 750, 1050 
    
    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, 'w', compression=zipfile.ZIP_DEFLATED) as out_zip:
        for i in range(0, len(imgs), 9):
            sheet = Image.new('RGB', (CW*3, CH*3), (255, 255, 255))
            chunk = imgs[i:i+9]
            for idx, (name, data) in enumerate(chunk):
                with Image.open(io.BytesIO(data)) as card_img:
                    card_img = card_img.resize((CW, CH), Image.Resampling.LANCZOS)
                    sheet.paste(card_img, ((idx % 3) * CW, (idx // 3) * CH))
            
            s_buf = io.BytesIO()
            sheet.save(s_buf, 'JPEG', quality=85)
            out_zip.writestr(f"Bogen_{i//9 + 1}.jpg", s_buf.getvalue())
            gc.collect()

    out_buf.seek(0)
    return send_file(out_buf, mimetype='application/zip', as_attachment=True, download_name='druck.zip')

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>MTG Tool v3.3</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #000; color: #0f0; padding-top: 50px; }
        .box { border: 2px solid #0f0; padding: 20px; display: inline-block; width: 300px; margin: 10px; }
        input[type="submit"] { background: #0f0; color: #000; border: none; padding: 10px; cursor: pointer; width: 100%; font-weight: bold; }
    </style>
    <h1>MTG Proxy Tool v3.3</h1>
    <div class="box">
        <h3>1. Split & Trim</h3>
        <form action="/split" method="post" enctype="multipart/form-data">
            <input type="file" name="file"><br><br>
            <input type="submit" value="START SPLIT">
        </form>
    </div>
    <div class="box">
        <h3>2. Merge 3x3</h3>
        <form action="/merge" method="post" enctype="multipart/form-data">
            <input type="file" name="file"><br><br>
            <input type="submit" value="START MERGE">
        </form>
    </div>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
