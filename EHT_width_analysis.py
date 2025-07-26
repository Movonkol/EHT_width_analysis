import cv2
import numpy as np
from itertools import groupby
import os
import csv


def find_eht_thickness_from_image(img, threshold=80, search_range=80, profile_fraction=0.7, num_samples=300):
    """
    Detects red fiducial markers and measures engineered heart tissue (EHT) thickness
    by sampling a line perpendicular to the marker axis.
    """
    # Convert to HSV and mask red markers
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Detect marker circles
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=10, minRadius=5, maxRadius=30)
    if circles is None or len(circles[0]) < 2:
        return None

    # Select two markers farthest apart
    points = [(int(x), int(y)) for x, y, r in circles[0]]
    max_dist = 0
    markers = None
    for p1 in points:
        for p2 in points:
            dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
            if dist > max_dist:
                max_dist = dist
                markers = (p1, p2)
    p1, p2 = markers

    # Compute perpendicular sampling line
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = np.hypot(dx, dy)
    perp = np.array([-dy / length, dx / length])
    center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

    thickness_values = []
    # Sample along perpendicular direction
    for shift in np.linspace(-search_range, search_range, num_samples):
        x = center[0] + perp[0] * shift
        y = center[1] + perp[1] * shift
        # Extract grayscale profile
        line = cv2.getRectSubPix(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            (1, int(length * profile_fraction)),
            (x, y)
        )
        arr = line.flatten()
        # Binary mask of tissue pixels
        bw = arr > threshold
        # Measure longest contiguous tissue block
        runs = [sum(1 for _ in group) for val, group in groupby(bw) if val]
        thickness = max(runs) if runs else 0
        thickness_values.append(thickness)

    return max(thickness_values)


def analyze_video(video_path, threshold=80, search_range=80, profile_fraction=0.7, max_seconds=7):
    """
    Processes a video to find minimum and maximum EHT thickness.
    Returns a tuple: (min_thickness, max_thickness, min_frame_file, max_frame_file).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * max_seconds)
    min_thick, max_thick = float('inf'), 0
    min_frame_file, max_frame_file = None, None

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        thickness = find_eht_thickness_from_image(
            frame, threshold, search_range, profile_fraction
        )
        if thickness is not None:
            base = os.path.splitext(os.path.basename(video_path))[0]
            if thickness < min_thick:
                min_thick = thickness
                min_frame_file = f"{base}_min_frame.png"
                cv2.imwrite(min_frame_file, frame)
            if thickness > max_thick:
                max_thick = thickness
                max_frame_file = f"{base}_max_frame.png"
                cv2.imwrite(max_frame_file, frame)
        frame_idx += 1

    cap.release()
    return min_thick, max_thick, min_frame_file, max_frame_file


if __name__ == "__main__":
    # Main execution
    folder_path = r"C:\path\to\EHT_videos"  # Adjust this path
    video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.avi', '.mp4', '.mov'))]
    results = []

    for video in video_files:
        print(f"Analyzing {video}...")
        min_t, max_t, min_file, max_file = analyze_video(
            video, threshold=40, search_range=90, profile_fraction=0.7, max_seconds=7
        )
        results.append([os.path.basename(video), min_t, max_t, min_file, max_file])

    # Save results
    with open('EHT_analysis_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Video', 'Min_Thickness_px', 'Max_Thickness_px', 'Min_Frame', 'Max_Frame'])
        writer.writerows(results)

    print("Results have been saved to 'EHT_analysis_results.csv'.")
