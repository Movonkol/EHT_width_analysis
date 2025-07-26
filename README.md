# EHT Analysis Script

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.7%2B-yellow.svg)](https://www.python.org/)

Automated analysis of Engineered Heart Tissue (EHT) thickness from video recordings using red fiducial marker detection and intensity profiling.

## Features

* **Automatic Marker Detection**: Detects two red fiducial markers per frame (HSV + Hough Circle).
* **Perpendicular Profiling**: Samples grayscale intensity along a line perpendicular to markers.
* **Thickness Extraction**: Identifies the longest contiguous tissue region per frame.
* **Min/Max Analysis**: Determines minimum (relaxed) and maximum (contracted) thickness across a time window.
* **Batch Processing**: Processes all videos in a directory and compiles results.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/eht-analysis.git
   cd eht-analysis
   ```
2. Install dependencies:

   ```bash
   pip install opencv-python numpy
   ```

## Configuration

Edit the `folder_path` variable in `EHT_width_analysis.py` to point to your video directory:

```python
folder_path = r"C:\path\to\EHT_videos"
```

Optional parameters (in `analyze_video` and `find_eht_thickness_from_image`):

* `threshold` (int): Grayscale cutoff (default: 80)
* `search_range` (int): Sampling range in px (default: 80)
* `profile_fraction` (float): Profile length fraction (default: 0.7)
* `max_seconds` (int): Duration to analyze per video (default: 7)
* `num_samples` (int): Shifts along profile (default: 300)

## Usage

Run the analysis script:

```bash
python EHT_width_analysis.py
```

## Output

* **Annotated Frames**: `<basename>_min_frame.png`, `<basename>_max_frame.png`
* **Results CSV**: `EHT_analysis_results.csv` (aggregated data)

## CSV Format

| Video              | Min\_Thickness\_px | Max\_Thickness\_px | Min\_Frame                     | Max\_Frame                     |
| ------------------ | ------------------ | ------------------ | ------------------------------ | ------------------------------ |
| example\_video.mp4 | 45                 | 72                 | example\_video\_min\_frame.png | example\_video\_max\_frame.png |

## Contributing

Contributions are welcome! Please submit issues or pull requests via GitHub.


