import cv2
import os
import cairosvg

# Setze den Pfad zum Ordner, der die Diagramme enthält
path = 'C:/Users/TGerb/Desktop/Studienarbeit/Python/HTMCFD/Diagramm'

# Setze den Namen des Ausgabevideos
video_name = os.path.join(path, "column_1_u_lateral_of.mp4")

# Definiere die Framerate des Ausgabevideos
fps = 1.0

# Definiere die Größe des Ausgabevideos
frame_size = (800, 600)

# Erstelle eine Liste der Diagramme im Ordner und sortiere sie numerisch
diagramme = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.svg')]

def extract_number(filename):
    basename = os.path.basename(filename)
    return int(basename.replace('sim_pred_column_1_u_65_65_', '').split('.')[0])

# Sortiere die Diagramme in der richtigen Reihenfolge
diagramme.sort(key=extract_number)

# Erstelle das Ausgabevideo
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

# Iteriere über jede SVG-Datei im Ordner
for diagramm in diagramme:
    # Konvertiere die SVG-Datei in eine temporäre PNG-Datei
    temp_file = "temp.png"
    cairosvg.svg2png(url=diagramm, write_to=temp_file)
    # Lade die temporäre PNG-Datei
    img = cv2.imread(temp_file)
    # Ändere die Größe des Bildes auf die gewünschte Größe
    img = cv2.resize(img, frame_size)
    # Schreibe das Bild in das Ausgabevideo
    out.write(img)
    # Lösche die temporäre PNG-Datei
    os.remove(temp_file)

# Schließe das Ausgabevideo
out.release()
