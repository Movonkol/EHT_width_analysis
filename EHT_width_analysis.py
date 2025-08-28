import cv2
import numpy as np
from itertools import groupby
import os
import csv
import math

# ======================
# Parameter (anpassen)
# ======================
FOLDER_PATH = r"C:\Users\Moritz\Downloads\EHT_mat.8\EHT_mat.8"  # <-- Pfad anpassen
THRESHOLD = 40            # Grauwert-Schwelle fürs Gewebe
SEARCH_RANGE = 90         # Pixel halbseitig entlang der Senkrechten
PROFILE_FRACTION = 0.7    # Anteil der Marker-Distanz für Profillänge
NUM_SAMPLES = 300         # Abtastpunkte entlang der Senkrechten
MAX_SECONDS = 5           # Wie viele Sekunden pro Video analysieren
MIN_MARKER_RADIUS = 4     # Marker-Filter (äquivalenter Radius aus Konturfläche)
MAX_MARKER_RADIUS = 60

# ======================
# Hilfsfunktionen
# ======================

def detect_red_markers(img_bgr):
    """
    Findet rote Marker (HSV + Morphologie + Konturen).
    Gibt zwei Zentren (am weitesten getrennt) zurück oder None.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # zwei Rotbereiche
    lower1, upper1 = np.array([0, 70, 50]),  np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # Glätten + Morphologie
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        r_eq = math.sqrt(area / math.pi)
        if r_eq < MIN_MARKER_RADIUS or r_eq > MAX_MARKER_RADIUS:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    if len(centers) < 2:
        return None

    # wähle die beiden am weitesten entfernten
    max_d, pair = -1, None
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            (x1, y1), (x2, y2) = centers[i], centers[j]
            d = math.hypot(x2 - x1, y2 - y1)
            if d > max_d:
                max_d, pair = d, (centers[i], centers[j])
    return pair  # ((x1,y1),(x2,y2))


def find_eht_thickness_from_image(img_bgr, threshold=80, search_range=80, profile_fraction=0.7, num_samples=300):
    """
    Misst EHT-Dicke (in Pixeln) senkrecht zur Markerachse.
    Zeichnet einen Messstrich exakt in EHT-Länge und schreibt die Dicke oben links.
    Rückgabe: (thickness_px:int|None, overlay_bgr:np.ndarray|None)
    """
    # --- Marker finden ---
    markers = detect_red_markers(img_bgr)
    if markers is None:
        return None, None
    (x1, y1), (x2, y2) = markers

    dx, dy = (x2 - x1), (y2 - y1)
    axis_len = float(np.hypot(dx, dy))
    if axis_len < 1:
        return None, None

    # Senkrechte (Einheitsvektor) und Mitte
    perp = np.array([-dy / axis_len, dx / axis_len], dtype=float)
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
    profile_len = int(max(5, axis_len * profile_fraction))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # längste True-Sequenz inkl. Indizes
    def longest_true_run_with_idx(bw):
        best_len, best_s, cur_s = 0, None, None
        cur_len = 0
        for i, v in enumerate(bw):
            if v:
                if cur_len == 0:
                    cur_s = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_s = cur_len, cur_s
            else:
                cur_len = 0
        if best_len == 0:
            return 0, None, None
        return best_len, best_s, best_s + best_len  # [start, end)

    # Suche über Verschiebungen: merke beste Stelle + exakte Endpunkte
    best = dict(len=0, cx=None, cy=None, y0=None, s=None, e=None)
    for shift in np.linspace(-search_range, search_range, num_samples):
        cx = center[0] + perp[0] * shift
        cy = center[1] + perp[1] * shift
        if not (0 <= cx < gray.shape[1] and 0 <= cy < gray.shape[0]):
            continue

        patch = cv2.getRectSubPix(gray, (1, profile_len), (float(cx), float(cy)))
        arr = patch.flatten()
        bw = arr > threshold

        run_len, s, e = longest_true_run_with_idx(bw)
        if run_len > best["len"]:
            y0 = cy - (profile_len - 1) / 2.0  # oberer Index des Profils im Bild
            best.update(len=int(run_len), cx=float(cx), cy=float(cy),
                        y0=float(y0), s=int(s) if s is not None else None,
                        e=int(e) if e is not None else None)

    if best["len"] <= 0 or best["s"] is None:
        return None, None

    # Endpunkte im Bildraum (clippen)
    x = int(round(best["cx"]))
    y_start = int(round(best["y0"] + best["s"]))
    y_end   = int(round(best["y0"] + best["e"] - 1))
    h = gray.shape[0]
    y_start = int(np.clip(y_start, 0, h - 1))
    y_end   = int(np.clip(y_end,   0, h - 1))

    thickness = best["len"]

    # --- Overlay zeichnen ---
    overlay = img_bgr.copy()
    # Marker + Marker-Achse
    cv2.circle(overlay, (x1, y1), 6, (0, 255, 0), 2)
    cv2.circle(overlay, (x2, y2), 6, (0, 255, 0), 2)
    cv2.line(overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Messstrich NUR über das EHT (BGR: (255,0,0) = Blau)
    cv2.line(overlay, (x, y_start), (x, y_end), (255, 0, 0), 2)

    # Text oben links
    cv2.putText(overlay, f"thickness={thickness} px", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return thickness, overlay


def analyze_video(video_path, threshold=80, search_range=80, profile_fraction=0.7, num_samples=300, max_seconds=7):
    """
    Durchläuft ein Video, misst pro Frame die Dicke und merkt sich min/max inkl. Overlay.
    Rückgabe: (min_thick:int|None, max_thick:int|None, min_file:str|None, max_file:str|None)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # Fallback
    max_frames = int(fps * max_seconds)

    base = os.path.splitext(os.path.basename(video_path))[0]
    min_thick, max_thick = None, None
    min_frame_file, max_frame_file = None, None

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        thickness, overlay = find_eht_thickness_from_image(
            frame, threshold=threshold, search_range=search_range,
            profile_fraction=profile_fraction, num_samples=num_samples
        )

        if thickness is not None:
            if (min_thick is None) or (thickness < min_thick):
                min_thick = thickness
                min_frame_file = f"{base}_min_overlay.png"
                cv2.imwrite(min_frame_file, overlay if overlay is not None else frame)
            if (max_thick is None) or (thickness > max_thick):
                max_thick = thickness
                max_frame_file = f"{base}_max_overlay.png"
                cv2.imwrite(max_frame_file, overlay if overlay is not None else frame)

        frame_idx += 1

    cap.release()
    return min_thick, max_thick, min_frame_file, max_frame_file


# ======================
# Main
# ======================
if __name__ == "__main__":
    folder_path = FOLDER_PATH
    video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))]

    results = []
    for video in video_files:
        print(f"Analysiere {video} ...")
        min_t, max_t, min_file, max_file = analyze_video(
            video_path=video,
            threshold=THRESHOLD,
            search_range=SEARCH_RANGE,
            profile_fraction=PROFILE_FRACTION,
            num_samples=NUM_SAMPLES,
            max_seconds=MAX_SECONDS
        )
        results.append([
            os.path.basename(video),
            "" if min_t is None else int(min_t),
            "" if max_t is None else int(max_t),
            min_file or "",
            max_file or ""
        ])

    # CSV speichern
    with open('EHT_analysis_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Video', 'Min_Thickness_px', 'Max_Thickness_px', 'Min_Frame', 'Max_Frame'])
        writer.writerows(results)

    print("Fertig. Ergebnisse in 'EHT_analysis_results.csv'. Overlays als *_min_overlay.png / *_max_overlay.png gespeichert.")


