# PrevOccupAI mBAN - Human Activity Recognition Pipeline

A complete end-to-end pipeline for Human Activity Recognition (HAR) using muscleBAN (mBAN) accelerometer data. This project processes raw sensor data from OpenSignals files through to trained machine learning models with rigorous cross-validation and post-processing optimization.

## 🎯 Overview

This project implements a comprehensive HAR system that:

1. **Processes raw mBAN sensor data** from OpenSignals format
2. **Segments activities** into protocol-compliant sub-activities with subject-specific optimization
3. **Extracts features** using TSFEL (Time Series Feature Extraction Library)
4. **Trains ML models** using nested cross-validation with subject-independent testing
5. **Optimizes predictions** through post-processing techniques (majority vote, threshold tuning, heuristics)

### Supported Activities

**Main Classes:**
- **Sitting** (Class 0)
- **Standing** (Class 1) 
- **Walking** (Class 2)

**Sub-Activities:**
- Sitting: `sit`
- Standing: `still`, `conversing` (talking)
- Cabinets: `drink_coffee`, `moving_objects` (folders)
- Walking: `slow`, `medium`, `fast`
- Stairs: `up`, `down`

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW DATA PROCESSING                         │
│  OpenSignals Files → Segmentation → Cropped Segments (.npy)    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                            │
│  Segmented Data → TSFEL Features → Feature Files (.npy/.csv)   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING (HAR)                           │
│  Features → Nested CV → Model Selection → Production Model      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    POST-PROCESSING                               │
│  Raw Predictions → Optimization → Final Predictions             │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.12+
- Virtual environment (recommended)
- ~2GB free disk space for dependencies

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/eLbARROS13/PrevOccupAI_mBAN_only.git
cd PrevOccupAI_mBAN_only
```

2. **Create and activate virtual environment:**

**On Unix/Mac:**
```bash
python3 -m venv prevOccupAI_venv
source prevOccupAI_venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv prevOccupAI_venv
.\prevOccupAI_venv\bin\Activate.ps1
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies

Key packages:
- `numpy==1.26.4` - Numerical computing
- `pandas==2.1.4` - Data manipulation
- `scikit-learn==1.4.2` - Machine learning
- `tsfel==0.1.9` - Feature extraction
- `scipy==1.11.4` - Signal processing
- `matplotlib==3.7.5` - Visualization
- `tqdm==4.66.4` - Progress bars
- `joblib==1.3.2` - Model serialization

## 🚀 Quick Start

### 1. Process Raw Data

Edit `main_mban.py` to configure paths:

```python
RAW_DATA_FOLDER_PATH = '/path/to/your/raw/data'
OUTPUT_FOLDER_PATH = '/path/to/output/segmented/data'
CROP_SECONDS = 5  # Crop from beginning/end of segments
```

Run the segmentation pipeline:

```bash
python main_mban.py
```

**Expected output structure:**
```
OUTPUT_FOLDER_PATH/
└── segmented_data/
    ├── P001/
    │   ├── P001_sitting_file01_sit_GlobalSegment1.npy
    │   ├── P001_walking_file01_slow_LocalSegment1.npy
    │   └── ...
    ├── P002/
    └── ...
```

### 2. Extract Features

Edit `feature_extractor/feature_extractor_main.py`:

```python
extract_mban_features(
    data_path='/path/to/segmented/data',
    features_data_path='/path/to/output/features',
    activities=['sitting', 'walking', 'cabinets', 'stairs', 'standing'],
    fs=1000,           # mBAN sampling rate
    window_size=5,     # 5-second windows
    overlap=0.5        # 50% overlap
)
```

Run feature extraction:

```bash
python feature_extractor/feature_extractor_main.py
```

**Output:** 45 TSFEL features per window, saved per subject.

### 3. Train HAR Model

Configure `tmp_har_windows_package/har_training_windows/har_config.py`:

```python
DATA_PATH = '/path/to/extracted/features'
BALANCING_TYPE = 'main_classes'  # or 'sub_classes', None
WINDOW_SIZE_SAMPLES = 5000  # matches feature extraction
```

Run model training:

```bash
cd tmp_har_windows_package/har_training_windows
python har_pipeline.py
```

**This performs:**
- Nested cross-validation (5 outer × 2 inner folds)
- Subject-independent testing
- 3 algorithms × 3 normalizations × 7 feature counts = **63 configurations**
- Saves best model as `.joblib` file

### 4. Optimize with Post-Processing

Edit `postprocess_config.py`:

```python
MODEL_PATH = Path("path/to/your/model.joblib")
TEST_DATA_PATH = "path/to/test/data"
DATA_FORMAT = "csv"  # or "npz", "custom"
```

Run post-processing optimization:

```bash
python har_postprocess_pipeline.py
```

**Expected improvements:** +3-5% accuracy through majority vote, threshold tuning, and heuristics.

## 🔧 Pipeline Components

### 1. Raw Data Processing (`raw_data_processor/`)

#### **mBAN Data Segmenter** (`mban_data_segmenter.py`)

Segments mBAN OpenSignals files into protocol-compliant activities.

**Key features:**
- Subject-specific parameter optimization for problematic files
- Automatic post-processing for oversegmentation
- Protocol-compliant segment validation
- Enhanced segmentation for 24 problematic cases

**Segmentation methods:**
- **Peak detection** for standing/cabinet activities
- **Onset-based** for walking/stairs activities
- **Subject-specific tuning** for challenging signals

**Configuration:**
```python
MBAN_FS = 1000  # 1000 Hz sampling rate
CROP_SECONDS = 5  # Crop from start/end of each segment
```

**Enhanced features:**
- Systematic visual analysis-based parameter optimization
- Ultra-sensitive parameters for undersegmentation (P002, P004, P007, P012)
- Heavy smoothing for oversegmentation (P015, P016, P017, P020)
- Expected segment mapping per subject/activity

#### **Signal Processing** (`filters.py`, `interpolate.py`)

- Butterworth filtering for noise reduction
- Interpolation for missing data points
- Signal quality validation

### 2. Feature Extraction (`feature_extractor/`)

#### **Feature Extractor** (`feature_extractor.py`)

Extracts time-series features using TSFEL library.

**Features extracted (45 total):**
- **Temporal:** Mean, median, std, variance, min, max, range
- **Statistical:** Skewness, kurtosis, entropy, interquartile range
- **Spectral:** FFT coefficients, power spectral density
- **Complexity:** Zero-crossing rate, slope changes

**Windowing:**
```python
window_size = 5        # 5-second windows
overlap = 0.5          # 50% overlap
fs = 1000              # 1000 Hz sampling rate
```

**Flexible filename support:**
- New format: `P001_cabinets2_file03_moving_objects_GlobalSegment1.npy`
- Original format: `P001_cabinets_file01_drink_coffee.npy`
- Automatically filters `_failed.npy` files

**Activity mapping:**
```python
MBAN_SUB_ACTIVITY_MAP = {
    'sitting': 'sit',
    'still': 'still',
    'talk': 'talk',
    'objects': 'folders',  # moving_objects → folders
    'coffee': 'coffee',
    'slow': 'slow',
    'medium': 'medium',
    'fast': 'fast',
    'up': 'up',
    'down': 'down'
}
```

#### **Configuration** (`cfg_file.json`)

TSFEL feature configuration file. Customize which features to extract.

### 3. HAR Model Training (`HAR/`)

#### **Model Selection** (`model_selection.py`)

Performs comprehensive model evaluation and selection.

**Algorithms tested:**
1. **Random Forest**
   - Criteria: gini, entropy
   - Trees: 50, 100, 500, 1000
   - Max depth: None, 10, 20, 30
   
2. **Support Vector Machine (SVM)**
   - Kernels: rbf, linear
   - C: 0.0001 to 1000
   - Gamma: scale, auto, 0.001 to 10
   
3. **K-Nearest Neighbors (KNN)**
   - Neighbors: 1 to 14
   - Distance metrics: Manhattan (p=1), Euclidean (p=2)

**Normalization schemes:**
- None (raw features)
- MinMax scaling (0-1)
- Standard scaling (z-score)

**Feature selection:**
- Low variance removal
- Highly correlated feature removal
- K-best feature selection: [5, 10, 15, 20, 25, 30, 35]

#### **Cross-Validation** (`cross_validation.py`)

**Nested Cross-Validation Structure:**

```
Outer CV (5 folds - StratifiedGroupKFold)
│   Purpose: Unbiased model performance estimation
│   Stratification: Maintains class balance
│   Grouping: Subject IDs (subject-independent)
│
├── Fold 1
│   ├── Inner CV (2 folds - StratifiedGroupKFold)
│   │   Purpose: Hyperparameter optimization
│   │   Method: GridSearchCV
│   │   Groups: Subject IDs within training data
│   │
│   └── Test on validation fold → Accuracy
│
├── Fold 2
│   └── ...
...
└── Fold 5
    └── Final accuracy: Mean ± Std across all folds
```

**Key properties:**
- **Subject-independent:** Same subject never in train & validation
- **Stratified:** Maintains class distribution
- **Grouped:** Respects subject boundaries
- **Unbiased:** Separate hyperparameter tuning for each fold

**Total evaluations:** 630 model training cycles (3 algorithms × 3 normalizations × 7 feature counts × 5 outer folds × 2 inner folds)

#### **Feature Selection** (`feature_selection.py`)

Three-stage feature selection pipeline:
1. Remove low-variance features (< 0.01 threshold)
2. Remove highly correlated features (> 0.9 correlation)
3. Select K-best features using ANOVA F-test

#### **Data Loading** (`load.py`)

Handles various data formats:
- NumPy arrays (`.npy`)
- CSV files (`.csv`)
- Metadata JSON files

Performs train-test splitting with subject-independent grouping.

## 📁 Project Structure

```
PrevOccupAI_mBAN_only/
│
├── main_mban.py                     # Main pipeline entry point
├── constants.py                     # Global constants and mappings
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── raw_data_processor/              # Data segmentation module
│   ├── __init__.py
│   ├── mban_data_segmenter.py      # mBAN-specific segmentation
│   ├── segment_activities.py       # Activity segmentation logic
│   ├── load_sensor_data.py         # OpenSignals data loading
│   ├── filters.py                  # Signal filtering
│   ├── interpolate.py              # Data interpolation
│   └── pre_process.py              # Preprocessing utilities
│
├── feature_extractor/               # Feature extraction module
│   ├── __init__.py
│   ├── feature_extractor_main.py   # Main extraction script
│   ├── feature_extractor.py        # Core extraction logic
│   ├── quaternion_features.py      # Quaternion-specific features
│   ├── window.py                   # Windowing utilities
│   └── cfg_file.json               # TSFEL configuration
│
├── HAR/                            # Machine learning module
│   ├── __init__.py
│   ├── cross_validation.py         # Nested CV implementation
│   ├── model_selection.py          # Model training & selection
│   ├── feature_selection.py        # Feature selection methods
│   ├── load.py                     # Data loading utilities
│   ├── post_process.py             # Post-processing techniques
│   └── post_processing_optimizer.py # Optimization framework
│
├── tmp_har_windows_package/        # Standalone HAR package
│   └── har_training_windows/
│       ├── har_pipeline.py         # Complete training pipeline
│       ├── har_config.py           # Configuration file
│       └── windows_run_pipeline.bat # Windows batch script
│
├── docs/                           # Documentation
│   └── enhanced_segmentation_system.md
│
├── har_postprocess_pipeline.py     # Post-processing entry point
├── postprocess_config.py           # Post-processing configuration
├── data_loader_helper.py           # Data loading helpers
├── file_utils.py                   # File system utilities
│
└── prevOccupAI_venv/               # Virtual environment
    ├── bin/
    ├── lib/
    └── pyvenv.cfg
```

## 📄 License

This project is part of the PrevOccupAI research initiative.

## 📧 Contact

For questions or issues, please open an issue on the GitHub repository.
