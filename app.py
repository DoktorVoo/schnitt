import cv2
import numpy as np
import os
import shutil
import io
from flask import Flask, request, send_file, render_template_string
from werkzeug.utils import secure_filename
import zipfile

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
OUTPUT_FOLDER = 'temp_output'

# Einfaches HTML-Interface für den Upload
HTML_PAGE = '''
<!doctype html>
<title>MTG Card Extractor</title>
<h1>MTG Karten-Cutter</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=files multiple>
  <input type=submit value=Verarbeiten>
</form>
<p>Lade deine JPGs (auch Split-Bilder) hoch. Du erhältst eine ZIP zurück.</p>
'''

def sort_contours(rects):
    if not rects: return []
    rects.sort(key=lambda b: b[1])
    rows = []
    current_row = [rects[0]]
    tolerance = rects[0][3] // 2 
    for rect in rects[1:]:
        if abs(rect[1] - current_row[0][1]) < tolerance:
            current_row.append(rect)
        else:
            rows.append(current_row)
            current_row = [rect]
    rows.append(current_row)
    sorted_rects = []
    for row in rows:
        row.sort(key=lambda b: b[0])
        sorted_rects.extend(row)
    return sorted_rects

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
        if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
        
        files = request.files.getlist('files')
        for file in files:
            filename = secure_filename(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)
            
            # Bildverarbeitung
            img = cv2.imread(img_path)
            base_name = os.path.splitext(filename)[0]
            is_split = "Split" in base_name
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (5,5), 0), 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10000]
            sorted_rects = sort_contours(valid_rects)
            
            counter = 1
            for (x, y, w, h) in sorted_rects:
                if is_split:
                    half_w = w // 2
                    cv2.imwrite(f"{OUTPUT_FOLDER}/{base_name}_{counter}.jpg", img[y:y+h, x:x+half_w])
                    counter += 1
                    cv2.imwrite(f"{OUTPUT_FOLDER}/{base_name}_{counter}.jpg", img[y:y+h, x+half_w:x+w])
                    counter += 1
                else:
                    cv2.imwrite(f"{OUTPUT_FOLDER}/{base_name}_{counter}.jpg", img[y:y+h, x:x+w])
                    counter += 1

        # ZIP erstellen
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for root, dirs, files in os.walk(OUTPUT_FOLDER):
                for f in files:
                    zf.write(os.path.join(root, f), f)
        memory_file.seek(0)
        
        # Temp Ordner aufräumen
        shutil.rmtree(UPLOAD_FOLDER)
        shutil.rmtree(OUTPUT_FOLDER)
        
        return send_file(memory_file, download_name="MTG_Cards.zip", as_attachment=True)

    return HTML_PAGE

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
