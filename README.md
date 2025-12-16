# EV Tracker - Extracellular Vesicle Detection and Tracking

A simplified, object-oriented interface for analyzing extracellular vesicle (EV) movements in microscopy image sequences.

## Overview

EV Tracker provides automated detection, tracking, and analysis of extracellular vesicles in TIFF image stacks. The pipeline includes:

- **Background subtraction** - Temporal median filtering
- **Enhancement** - CLAHE contrast enhancement and noise reduction
- **Detection** - Template matching with correlation-based particle detection
- **Tracking** - Frame-to-frame particle linking with gap handling
- **Metrics** - Precision-recall curves, ROC analysis, and performance metrics

## Installation

1. Clone or download this repository

2. Install dependencies:
   ```bash
   cd UMB_EV_Tracker
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python -c "from src.ev_tracker import EVTracker; print('Ready!')"
   ```

## Important: Running Location

**All Python scripts must be run from the project root directory (`UMB_EV_Tracker/`), NOT from `src/`.**

```bash
# Correct
cd UMB_EV_Tracker/
python src/test_all_features.py

# Incorrect
cd UMB_EV_Tracker/src/
python test_all_features.py  # Import errors!
```

## Quick Start

```python
from src.ev_tracker import EVTracker

# Single file analysis
tracker = EVTracker()
tracker.set_params(threshold=0.55, min_distance=30)
results = tracker.run("movie.tif", "ground_truth.csv")
print(f"AP: {results['global_ap']:.3f}")

# Batch analysis (multiple files)
datasets = [("movie1.tif", "gt1.csv"), ("movie2.tif", "gt2.csv")]
results = tracker.run_batch(datasets)
print(f"Global AP: {results['global_ap']:.3f}")
```

## Understanding `run()` vs `run_batch()`

### CRITICAL DIFFERENCE

| Method | Threshold Behavior | Use When |
|--------|-------------------|----------|
| `run()` | Uses specified/default threshold | Single file, specific threshold, parameter tuning |
| `run_batch()` | Overrides to 0.1 | Multiple files, comprehensive PR curves |

### Why the Override?

To compute accurate **Precision-Recall curves**, we need ALL possible detections (even weak ones) with their confidence scores. The threshold is then varied post-hoc to generate the curve.

- If you detect only at threshold=0.6, you miss all 0.1-0.59 detections
- `run()` - Uses YOUR threshold for normal analysis
- `run_batch()` - Uses 0.1 to capture everything for comprehensive curves

### Examples

```python
# Example 1: Test a specific threshold
tracker.set_params(threshold=0.6)
results = tracker.run("movie.tif", "gt.csv")  # Uses 0.6

# Example 2: Comprehensive PR curves
tracker.set_params(threshold=0.6)  # Will be ignored!
results = tracker.run_batch([("movie.tif", "gt.csv")])  # Uses 0.1

# Example 3: Parameter sweep
for thresh in [0.4, 0.5, 0.6, 0.7]:
    tracker.set_params(threshold=thresh)
    results = tracker.run("movie.tif", "gt.csv")  # Each uses their thresh
    print(f"Threshold {thresh}: AP = {results['global_ap']:.3f}")
```

**Note:** All other parameters (min_distance, max_distance, etc.) are respected by BOTH methods.

## Parameters

### All Configurable Parameters

#### Detection (5 parameters)

| Parameter | Default | Description | Range |
|-----------|---------|-------------|-------|
| `threshold` | 0.55 | Detection confidence (0-1) | 0.4-0.7 |
| `min_distance` | 30 | Min separation between particles (px) | 20-40 |
| `filter_radius` | 10 | Expected particle radius (px) | 5-15 |
| `filter_size` | 41 | Filter matrix size (px, must be odd) | 21-61 |
| `filter_sigma` | 2.0 | Gaussian smoothing | 1.0-4.0 |

#### Background & Enhancement (4 parameters)

| Parameter | Default | Description | Range |
|-----------|---------|-------------|-------|
| `bg_window_size` | 15 | Temporal window (frames) | 5-30 |
| `blur_kernel_size` | 7 | Noise reduction (px, must be odd) | 3-15 |
| `clahe_clip_limit` | 2.0 | Contrast limit | 1.0-4.0 |
| `clahe_grid_size` | (8, 8) | Contrast tile size | (4,4)-(16,16) |

#### Tracking (3 parameters)

| Parameter | Default | Description | Range |
|-----------|---------|-------------|-------|
| `max_distance` | 25 | Max movement per frame (px) | 15-40 |
| `min_track_length` | 5 | Min frames to keep track | 3-15 |
| `max_frame_gap` | 3 | Max gap in track (frames) | 1-10 |

#### Metrics (1 parameter)

| Parameter | Default | Description | Range |
|-----------|---------|-------------|-------|
| `distance_threshold` | 30.0 | Evaluation distance (px) | 15-50 |

### Setting Parameters

```python
tracker = EVTracker()

# Set common parameters
tracker.set_params(threshold=0.55, min_distance=30, max_distance=25)

# Set all at once
tracker.set_params(
    threshold=0.55, min_distance=30, filter_radius=10,
    bg_window_size=15, blur_kernel_size=7,
    max_distance=25, min_track_length=5
)

# View current settings
tracker.print_params()
```

## Output

### Results Dictionary

```python
results = {
    'success': True,                # Whether completed
    'global_ap': 0.82,              # Average Precision
    'global_auc': 0.88,             # ROC AUC
    'total_points': 5432,           # Total detections
    'output_dir': 'path/to/out',    # Output directory
    'file_summaries': [...]         # Per-file metrics
}
```

### Sample Output Directory Structure

```
UMB_EV_Tracker/out/
├── global_metrics/                        # From run_batch()
│   └── run_TIMESTAMP/
│       ├── global_performance_curves.png  # PR & ROC plots
│       ├── global_pr_curve_data.csv
│       └── file_summaries.csv
└── ev_detection_results_TIMESTAMP/        # From run()
    ├── 02_filter_creation/
    ├── 03_background_subtraction/
    ├── 04_enhancement/
    ├── 05_detection/                      # Overlays, videos
    ├── 06_tracking/                       # Track visualizations
    └── 07_metrics/                        # CSVs, PR curves
```

### Key Output Files

- `*_all_detections.csv` - All detections with coordinates, confidence, track IDs
- `track_summaries.csv` - Per-track statistics (length, velocity, etc.)
- `threshold_analysis.csv` - Performance at different thresholds
- `global_performance_curves.png` - PR and ROC curves

## Performance Metrics Guide

| Metric | Excellent | Good | Poor | Action |
|--------|-----------|------|------|--------|
| **AP** | > 0.8 | 0.6-0.8 | < 0.6 | Tune threshold/parameters |
| **Position Error** | < 10px | 10-20px | > 20px | Check alignment/threshold |
| **Detection Rate** | > 75% | 50-75% | < 50% | Adjust threshold/tracking |

## Common Parameter Adjustments

```python
# More sensitive (finds dim particles, more false positives)
tracker.set_params(threshold=0.45)

# Less sensitive (fewer false positives, may miss dim particles)
tracker.set_params(threshold=0.65)

# Larger particles (~25px)
tracker.set_params(filter_radius=12, min_distance=40)

# Smaller particles (~10px)
tracker.set_params(filter_radius=5, min_distance=20)

# Fast-moving particles
tracker.set_params(max_distance=35, max_frame_gap=5)

# Slow-moving particles
tracker.set_params(max_distance=15, max_frame_gap=2)
```

## Ground Truth Format

CSV files must contain:

```csv
Slice,X_COM,Y_COM,EV_ID
1,123.4,456.7,1
2,125.1,458.3,1
3,126.8,460.2,1
```

- `Slice`: Frame number (1-indexed)
- `X_COM`: X coordinate (pixels)
- `Y_COM`: Y coordinate (pixels)
- `EV_ID` (optional): Particle ID

Make sure that there aren't any rows full of ",,,,,,,,,", that will mess up the pipeline.

## API Reference

### EVTracker Class

```python
EVTracker(output_dir="UMB_EV_Tracker/out")
```

#### Methods

**`set_params(**kwargs) -> EVTracker`**
- Set any of the 13 parameters
- Returns `self` for method chaining
- See Parameters section for all options

**`run(tiff_file, ground_truth_csv=None) -> Dict`**
- Analyzes single file **using YOUR threshold**
- Returns: Results dictionary

**`run_batch(dataset_list) -> Dict`**
- Analyzes multiple files
- **Overrides threshold to 0.1** for comprehensive PR curves
- `dataset_list`: List of (tiff_file, csv_file) tuples
- Returns: Aggregated global metrics

**`print_params()`**
- Displays current parameter settings

### Convenience Function

**`quick_analyze(tiff_file, ground_truth_csv=None, threshold=0.55) -> Dict`**

```python
from src.ev_tracker import quick_analyze
results = quick_analyze("movie.tif", "gt.csv", threshold=0.6)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named 'src'` | Run from project root: `cd UMB_EV_Tracker/` |
| Too many false positives | Increase threshold: `tracker.set_params(threshold=0.65)` |
| Missing particles | Decrease threshold: `tracker.set_params(threshold=0.45)` |
| Fragmented tracks | Increase movement/gaps: `tracker.set_params(max_distance=35, max_frame_gap=5)` |
| Memory issues | Reduce `bg_window_size`, process files individually |

## Testing

```bash
# Update file paths in src/test_all_features.py first
python src/test_all_features.py
```

## Project Structure

```
UMB_EV_Tracker/
├── src/
│   ├── ev_tracker.py          # Main API
│   ├── test_all_features.py   # Test suite
│   ├── helpers/               # Core pipeline
│   ├── metrics/               # Performance evaluation
│   └── pipeline/              # Image processing
├── data/                      # Input data
│   ├── tiff/                  # TIFF stacks
│   └── csv/                   # Ground truth
└── out/                       # Results
```

## Support

**Support:**
- GitHub Issues: [New Issue](https://github.com/shplok/UMB_EV_Tracker/issues/new)
- Email: s.bowerman.cs@gmail.com

---

**Version:** 1.0  
**Python:** 3.7+  
