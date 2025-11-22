# EHT Width Analysis – Installation & Usage Guide (VS Code friendly)

This repository contains a Python script that quantifies **minimum (relaxed)** and **maximum (contracted)** thickness of **Engineered Heart Tissues (EHTs)** from **video recordings** (DYNAQUBE) by detecting **two red fiducial markers** and sampling a perpendicular intensity profile through the tissue.

> **Input expectation:** Standard **video files** (e.g., `.mp4`, `.avi`). This only works with DYNAQUBE videos!

---

## 1) What you’ll do (quick overview)

1. Install **Python** and **VS Code**.
2. Put the script `EHT_width_analysis.py` into a project folder and open that folder in VS Code.
3. Use the **VS Code Terminal** only to run `pip install` for the dependencies.
4. Edit the **user settings** inside `EHT_width_analysis.py` (at least your `folder_path` to the videos).
5. Click the **green “Run” button** in VS Code to start the analysis (no need to type `python ...`).

---

## 2) Requirements

### 2.1 Operating system
- Windows 10/11 (examples below use Windows paths)
- macOS / Linux are also fine.

### 2.2 Software
- **Python 3.9+**
  - Download from the official Python website.
  - On Windows, during install, tick **“Add Python to PATH”**.
  - Verify:
    ```bash
    python --version
    ```
- **Visual Studio Code (VS Code)**
  - Download from the official VS Code website.
  - Install the **Python** extension (by Microsoft) via the Extensions tab.

### 2.3 Python packages
Open the built-in **VS Code Terminal** (Terminal → New Terminal) and install:

```bash
pip install opencv-python numpy
```

> If you are using a **virtual environment**, activate it first (see below).

---

## 3) Video formats & conversion (important)

The script expects **standard video containers/codecs** that OpenCV can read reliably (e.g., `.mp4` H.264, `.avi`).
---

## 4) Put the code into VS Code

### Copy/paste (simple)
1. Create a folder, e.g. `C:\Users\YourName\Documents\EHT_width_analysis`
2. Open this folder in VS Code: **File → Open Folder...**
3. Create a new file **`EHT_width_analysis.py`**.
4. Paste the script’s code into it and save (`Ctrl + S`).

---

## 5) (Optional) Create a virtual environment

Keeping dependencies isolated is recommended:

```bash
python -m venv .venv
```

Activate it:
- **PowerShell**:
  ```bash
  .venv\Scripts\Activate.ps1
  ```
- **cmd**:
  ```bash
  .venv\Scripts\activate.bat
  ```
- **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

Then install packages inside this environment:
```bash
pip install opencv-python numpy
```
---

## 7) Configure the script (user settings)

Open `EHT_width_analysis.py` and adjust the **user-editable variables** at the top (or where indicated in the file). At minimum set:

```python
folder_path = r"C:\Users\YourName\Documents\EHT_width_analysis\videos"
# results will be written next to the script or into a configured output folder if present
```

Common optional parameters (names may differ slightly depending on the current script version):

- `threshold` *(int)* – grayscale cutoff for profile segmentation (default ~80)
- `search_range` *(int)* – sampling range in pixels around the profile (default ~80)
- `profile_fraction` *(float)* – fraction of the marker distance to profile (default ~0.7)
- `max_seconds` *(int)* – maximum video duration to analyze (default ~7)
- `num_samples` *(int)* – number of profile shifts (default ~300)

> Internally, the script detects **two red circular markers** (HSV filter + Hough Circle) per frame, builds a **perpendicular** intensity profile through the EHT, identifies the **longest contiguous tissue band**, and aggregates **min/max thickness** across a time window.

Save the file after changes (`Ctrl + S`).

---

## 8) Run from VS Code

1. Open `EHT_width_analysis.py` in the editor.
2. Click the **green triangle** (“Run Python File”) at the top-right of the editor.  
   – or – **Run → Run Without Debugging** (`Ctrl + F5`).
3. If prompted, choose the correct Python interpreter (your `.venv` if you created one).

The script will iterate through all videos in `folder_path` and produce outputs.

---

## 9) Output

- **Annotated frames** for the detected **minimum** and **maximum** thickness:  
  - `<basename>_min_frame.png`  
  - `<basename>_max_frame.png`
- **Aggregated CSV** (default name): `EHT_analysis_results.csv`  
  Columns typically include: `Video`, `Min_Thickness_px`, `Max_Thickness_px`, `Min_Frame`, `Max_Frame`

Placeholders/example:

```text
Video,Min_Thickness_px,Max_Thickness_px,Min_Frame,Max_Frame
example_video.mp4,45,72,example_video_min_frame.png,example_video_max_frame.png
```

> Thickness is reported in **pixels**. If you need **µm**, multiply by your optical **pixel size (µm/px)** from calibration.

---

## 10) Tips for robust detection

- Use videos with **good, stable lighting** and **clear red fiducials**.
- Ensure **both markers are visible** and not occluded.
- Keep **camera motion** minimal; crop away moving borders if necessary.
- If markers aren’t detected:
  - Reduce compression (export at higher quality).
  - Adjust marker **HSV thresholds** or Hough-Circle parameters in the script (if exposed).
  - Increase `max_seconds` if your contraction window is longer.

---

## 11) Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'`**  
→ Install OpenCV:
```bash
pip install opencv-python
```

**Video fails to open or only a few frames are read**  
→ Convert/re-encode the video to a standard format:
```bash
ffmpeg -i input.ext -c:v libx264 -crf 18 -preset veryslow -pix_fmt yuv420p output.mp4
```
Try `.avi` if `.mp4` is problematic in your environment.

**Markers not detected / wrong region profiled**  
→ Check color fidelity (no grayscale). Tweak thresholds in the script. Improve lighting or marker visibility.

**CSV not created**  
→ Make sure there’s at least one valid video in `folder_path` and that the script has write permissions to the working directory.




