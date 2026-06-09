# Fetal Heart Rate (FHR) Extraction Tool

This repository implements a multi-stage digital signal processing pipeline for extracting Fetal Heart Rate (FHR) from Transabdominal Pulse Oximetry (TFO) data. The methodology is based on the 2021 paper: *"Multi-Detector Heart Rate Extraction Method for Transabdominal Fetal Pulse Oximetry"* (Kasap et al.). It was implemented by Shawn Feezer and Anusheel Nand with support from the Gemini and Claude LLM tools. 

## 🚀 Pipeline Overview

The extraction process is divided into five distinct stages to move from raw, noise-heavy electrical signals to a clean, fused heart rate estimate.

1.  **Stage 1: Preprocessing** – Applies a zero-phase Butterworth bandpass filter (0.2–15Hz) to remove baseline drift and high-frequency noise.
2.  **Stage 2: Adaptive Noise Cancellation (ANC)** – Uses the shallowest detector (D1) as a maternal reference. A Recursive Least Squares (RLS) filter adaptively subtracts maternal interference from deeper channels.
3.  **Stage 3: Spectral Estimation** – Segments the signal into 60s windows and computes the Power Spectral Density (PSD) using a 100th-order Yule-Walker Autoregressive (AR) model.
4.  **Stage 4: Peak Selection** – Identifies the dominant peak within the physiological range (110–270 BPM) for each detector and wavelength.
5.  **Stage 5: Sensor Fusion** – Combines 8 independent tracks (4 detectors × 2 wavelengths) using a weighted Median Absolute Deviation (MAD) outlier rejection and weighted mean.

---

## 🛠️ Usage

The master script `eec284.py` orchestrates the entire pipeline via a CLI.

### Basic Execution
```bash
python eec284.py --data_path data/PPG1.csv
```

### Visualization Mode
To see validation graphs for the final results:
```bash
python eec284.py --data_path data/PPG1.csv --graph
```
Mid-execution stage graphs can be manually toggled to true in `eec284.py` stage calls, but this will interupt flow until the graph is dismissed.

---

## 🎛️ Tunable Parameters

The system is highly configurable via CLI arguments:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--f_low` | 0.2 | Lower corner frequency for Stage 1 filter (Hz) |
| `--f_high` | 15.0 | Upper corner frequency for Stage 1 filter (Hz) |
| `--rls_taps` | 100 | Number of taps for the Stage 2 RLS filter |
| `--rls_lambda`| 0.9999| Forgetting factor for RLS (adaptivity speed) |
| `--window_size`| 60 | Duration of sliding spectrum windows (seconds) |
| `--window_step`| 30 | Update rate/step size of sliding windows (seconds) |
| `--yw_order` | 100 | Yule-Walker model order (spectral resolution) |
| `--bpm_min` | 110 | Minimum heart rate threshold for peak picking |
| `--bpm_max` | 270 | Maximum heart rate threshold for peak picking |
| `--s4_helper` | False | Enable stateful tracking (holds last value on signal loss) |
| `--mad_thresh` | 3.0 | Outlier rejection envelope multiplier for Stage 5 |
| `--w2`, `--w3` | 1.0, 3.0| Spatial reliability weights for Detectors 2 and 3 |

---

## 📂 File Directory

### Core Pipeline
*   **`eec284.py`**: The main entry point. Handles CLI parsing, data loading, and stage coordination.
*   **`stage1_preprocess.py`**: Implementation of Butterworth bandpass filtering.
*   **`stage2_anc.py`**: Implementation of the Recursive Least Squares (RLS) adaptive filter.
*   **`stage3_psd.py`**: Windowing logic and Yule-Walker spectral estimation.
*   **`stage4_peaks.py`**: Logic for extracting BPM values from PSD power maps.
*   **`stage5_fusion.py`**: Weighted sensor fusion and MAE/RMSE metric calculation.
*   **`plotting.py`**: Centralized utility for all spectral and time-domain visualizations.

### Research & Optimization
*   **`iterator_stage*.py`**: Scripts that run hundreds of iterations of the pipeline to "sweep" through different parameter settings.
*   **`analysis_*.py`**: Statistical visualization scripts that read the sweep results and generate bar charts to identify the best parameters.
*   **`master_sweep.py`**: A batch script to run all parameter optimizations in one command.
*   **`all_around_finder.py`**: Cross-dataset analysis to find the "best-fit" parameters that work across all patient data.

---

## 📊 Validation
The pipeline is validated against **1Hz Ground Truth hemodynamics** (the true fetal heart rate). After execution, the tool prints:
*   **MAE (Mean Absolute Error)**: Average BPM deviation.
*   **RMSE (Root Mean Squared Error)**: Accuracy metric that penalizes large tracking errors.

The final output plot is saved as `stage5_validation.png`.
# FHR_extraction_EEC284
Implementation and evaluation of, and experiments on Multi-Detector Heart Rate Extraction Method for Transabdominal Fetal Pulse Oximetry (2021)

tested with:
py eec284.py --data_path data\PPG1.csv

many flags for knob-turning are baked in.
