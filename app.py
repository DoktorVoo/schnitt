import cv2
import numpy as np
import os
import shutil
from glob import glob

# --- Konfiguration ---
INPUT_DIR = "input"       # Ordner, in dem deine zusammengefassten Bilder liegen
OUTPUT_DIR = "output"     # Ordner für die ausgeschnittenen Einzelkarten
ZIP_NAME = "Ausgeschnittene_Karten" # Name der finalen ZIP-Datei (ohne .zip)

def create_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    else:
        # Ordner leeren, falls er schon existiert
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))

def sort_contours(rects):
    """
    Sortiert die gefundenen Rechtecke zeilenweise von oben nach unten 
    und innerhalb der Zeile von links nach rechts.
    """
    if not rects:
        return []
    
    # Sortiere zuerst grob nach der Y-Achse (von oben nach unten)
    rects.sort(key=lambda b: b[1])
    
    rows = []
    current_row = [rects[0]]
    
    # Toleranz, um zu erkennen, ob Karten in der gleichen Zeile liegen (halbe Kartenhöhe)
    tolerance = rects[0][3] // 2 
    
    for rect in rects[1:]:
        if abs(rect[1] - current_row[0][1]) < tolerance:
            current_row.append(rect)
        else:
            rows.append(current_row)
            current_row = [rect]
            tolerance = rect[3] // 2
    rows.append(current_row)
    
    sorted_rects = []
    for row in rows:
        # Sortiere jede Zeile nach der X-Achse (von links nach rechts)
        row.sort(key=lambda b: b[0])
        sorted_rects.extend(row)
        
    return sorted_rects

def process_image(img_path):
    print(f"Verarbeite: {img_path}")
    img = cv2.imread(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    is_split = "Split" in base_name
    
    # In Graustufen umwandeln und leicht weichzeichnen, um Artefakte zu glätten
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding: Den weißen Hintergrund identifizieren. 
    # Alles was nicht strahlend weiß ist (schwarze Ränder), wird als Objekt erkannt.
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Konturen finden (RETR_EXTERNAL ignoriert Details im Inneren der Karte, wir wollen nur den äußeren Rand)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_rects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Ignoriere winzige Punkte/Staub (Wert anpassen, falls Karten ignoriert werden)
        if area > 10000: 
            x, y, w, h = cv2.boundingRect(contour)
            valid_rects.append((x, y, w, h))
            
    # Richtig sortieren (1_1, 1_2, 1_3 etc.)
    sorted_rects = sort_contours(valid_rects)
    
    counter = 1
    for (x, y, w, h) in sorted_rects:
        if is_split:
            # Bei Split-Karten ist das gefundene Rechteck eine Doppkarte. 
            # Wir schneiden sie exakt in der Mitte vertikal durch.
            half_w = w // 2
            
            # Linke Karte
            card1 = img[y:y+h, x:x+half_w]
            out_path1 = os.path.join(OUTPUT_DIR, f"{base_name}_{counter}.jpg")
            cv2.imwrite(out_path1, card1)
            counter += 1
            
            # Rechte Karte
            card2 = img[y:y+h, x+half_w:x+w]
            out_path2 = os.path.join(OUTPUT_DIR, f"{base_name}_{counter}.jpg")
            cv2.imwrite(out_path2, card2)
            counter += 1
            
        else:
            # Normale Karte
            card = img[y:y+h, x:x+w]
            out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{counter}.jpg")
            cv2.imwrite(out_path, card)
            counter += 1

def main():
    create_dirs()
    
    # Alle JPGs im input Ordner finden
    image_files = glob(os.path.join(INPUT_DIR, "*.jpg"))
    
    if not image_files:
        print(f"Keine JPG-Bilder im Ordner '{INPUT_DIR}' gefunden!")
        return

    for img_path in image_files:
        process_image(img_path)
        
    print("\nAlle Bilder verarbeitet. Erstelle ZIP-Archiv...")
    
    # Alles zippen
    shutil.make_archive(ZIP_NAME, 'zip', OUTPUT_DIR)
    print(f"Fertig! Das Archiv '{ZIP_NAME}.zip' wurde erstellt.")

if __name__ == "__main__":
    main()
