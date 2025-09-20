import cv2
import numpy as np
from itertools import groupby
import os
import csv
import math

# ======================
# Parameter (anpassen)
# ======================
FOLDER_PATH = r"C:\Users\Moritz\Downloads\EHT_Maturation.10\EHT_Maturation.10" # <-- Pfad anpassen
THRESHOLD = 40             # Grauwert-Schwelle fürs Gewebe
SEARCH_RANGE = 90          # Pixel halbseitig entlang der Senkrechten
PROFILE_FRACTION = 0.7     # Anteil der Marker-Distanz für Profillänge (für Dicke)
NUM_SAMPLES = 300          # Abtastpunkte entlang der Senkrechten
MAX_SECONDS = 5            # Wie viele Sekunden pro Video analysieren
MIN_MARKER_RADIUS = 4      # Marker-Filter (äquivalenter Radius aus Konturfläche)
MAX_MARKER_RADIUS = 60

# --- Kalibrierung: Pixel → Millimeter ---
# Annahme aus deinem Datensatz: 150 px = 3.0 mm  ⇒  1 px = 0.02 mm
CALIB_PX = 150.0
CALIB_MM = 3.0
MM_PER_PX = CALIB_MM / CALIB_PX  # = 0.02 mm/px

# Format für Overlay (z.B. 2 Nachkommastellen in mm)
MM_DECIMALS = 2

# ======================
# Hilfsfunktionen
# ======================

def px_to_mm(val_px: float | int | None) -> float | None:
    if val_px is None:
        return None
    return float(val_px) * MM_PER_PX


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


def find_eht_thickness_and_length_from_image(img_bgr, threshold=80, search_range=80, profile_fraction=0.7, num_samples=300):
    """
    Misst EHT-Dicke (in Pixeln) senkrecht zur Markerachse und berechnet zusätzlich die EHT-Länge
    (Marker-zu-Marker-Distanz in Pixeln).
    Zeichnet Messstrich(e) und schreibt Dicke & Länge inkl. mm-Umrechnung ins Overlay.
    Rückgabe: (thickness_px:int|None, length_px:int|None, overlay_bgr:np.ndarray|None)
    """
    # --- Marker finden ---
    markers = detect_red_markers(img_bgr)
    if markers is None:
        return None, None, None
    (x1, y1), (x2, y2) = markers

    dx, dy = (x2 - x1), (y2 - y1)
    axis_len = float(np.hypot(dx, dy))
    if axis_len < 1:
        return None, None, None

    # EHT-Länge als Marker-zu-Marker-Distanz (Pixel)
    eht_length_px = int(round(axis_len))

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

    # Suche über Verschiebungen: merke beste Stelle + exakte Endpunkte (für Dicke)
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

    # --- Overlay zeichnen ---
    overlay = img_bgr.copy()
    # Marker + Marker-Achse
    cv2.circle(overlay, (x1, y1), 6, (0, 255, 0), 2)
    cv2.circle(overlay, (x2, y2), 6, (0, 255, 0), 2)
    cv2.line(overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)

    thickness = None
    if best["len"] > 0 and best["s"] is not None:
        x = int(round(best["cx"]))
        y_start = int(round(best["y0"] + best["s"]))
        y_end   = int(round(best["y0"] + best["e"] - 1))
        h = gray.shape[0]
        y_start = int(np.clip(y_start, 0, h - 1))
        y_end   = int(np.clip(y_end,   0, h - 1))

        thickness = int(best["len"])
        # Messstrich NUR über das EHT (BGR: (255,0,0) = Blau)
        cv2.line(overlay, (x, y_start), (x, y_end), (255, 0, 0), 2)

    # --- Text oben links (nur mm) ---
    len_mm = px_to_mm(eht_length_px)
    if thickness is None:
        cv2.putText(overlay, f"length={len_mm:.{MM_DECIMALS}f} mm", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        thick_mm = px_to_mm(thickness)
        cv2.putText(overlay, f"thickness={thick_mm:.{MM_DECIMALS}f} mm", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"length={len_mm:.{MM_DECIMALS}f} mm", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return thickness, eht_length_px, overlay


def analyze_video(video_path, threshold=80, search_range=80, profile_fraction=0.7, num_samples=300, max_seconds=7):
    """
    Durchläuft ein Video, misst pro Frame die Dicke (min/max + Overlays) und ermittelt die Median-Länge.
    Rückgabe: (min_thick_px:int|None, max_thick_px:int|None, median_len_px:int|None, min_file:str|None, max_file:str|None)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # Fallback
    max_frames = int(fps * max_seconds)

    base = os.path.splitext(os.path.basename(video_path))[0]
    min_thick, max_thick = None, None
    min_frame_file, max_frame_file = None, None
    lengths = []

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        thickness, length_px, overlay = find_eht_thickness_and_length_from_image(
            frame, threshold=threshold, search_range=search_range,
            profile_fraction=profile_fraction, num_samples=num_samples
        )

        # Länge sammeln, wenn Marker gefunden
        if length_px is not None:
            lengths.append(int(length_px))

        # Dicke: min/max + Overlays schreiben
        if thickness is not None:
            if (min_thick is None) or (thickness < min_thick):
                min_thick = int(thickness)
                min_frame_file = f"{base}_min_overlay.png"
                cv2.imwrite(min_frame_file, overlay if overlay is not None else frame)
            if (max_thick is None) or (thickness > max_thick):
                max_thick = int(thickness)
                max_frame_file = f"{base}_max_overlay.png"
                cv2.imwrite(max_frame_file, overlay if overlay is not None else frame)

        frame_idx += 1

    cap.release()

    median_len = int(np.median(lengths)) if len(lengths) > 0 else None
    return min_thick, max_thick, median_len, min_frame_file, max_frame_file


# ======================
# Main
# ======================
if __name__ == "__main__":
    print(f"Kalibrierung: {CALIB_PX:.0f} px = {CALIB_MM:.3f} mm  (=> {MM_PER_PX:.5f} mm/px)")

    folder_path = FOLDER_PATH
    video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))]

    results = []
    for video in video_files:
        print(f"Analysiere {video} ...")
        min_t_px, max_t_px, median_len_px, min_file, max_file = analyze_video(
            video_path=video,
            threshold=THRESHOLD,
            search_range=SEARCH_RANGE,
            profile_fraction=PROFILE_FRACTION,
            num_samples=NUM_SAMPLES,
            max_seconds=MAX_SECONDS
        )

        # px → mm umrechnen
        min_t_mm = px_to_mm(min_t_px)
        max_t_mm = px_to_mm(max_t_px)
        median_len_mm = px_to_mm(median_len_px)

        results.append([
            os.path.basename(video),
            "" if median_len_mm is None else round(median_len_mm, 3),
            "" if min_t_mm is None else round(min_t_mm, 3),
            "" if max_t_mm is None else round(max_t_mm, 3),
            min_file or "",
            max_file or ""
        ])

    # CSV speichern (mit zusätzlichen mm-Spalten)
    with open('EHT_analysis_results_mm.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Video',
            'Median_Length_mm',
            'Min_Thickness_mm',
            'Max_Thickness_mm',
            'Min_Frame', 'Max_Frame'
        ])
        writer.writerows(results)

    print("Fertig. Ergebnisse in 'EHT_analysis_results_mm.csv'. Overlays zeigen nur mm.")
