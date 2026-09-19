# Pankaj Govekar — Project Worksheet

- **Name:** Pankaj Govekar
- **Role in Team:** Core Machine Learning, Data Preprocessing & Pipeline Architecture
- **Current Status:** 🟢 Downloaded & preprocessed PhysioNet Sleep-EDF dataset (42,308 epochs across 20 subjects). Ready for model training.
- **Next Step:** Push branch to GitHub & set up PyTorch CNN-Transformer training loop on the preprocessed data.
- **Last Updated:** 2026-09-19

---

## 📋 What I Have Done

| Date | Task / What I Did | How I Did It / Files Changed | Results & Verified Output | Status | What I Will Do Next |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2026-09-19** | **Downloaded Clinical EEG Dataset** | Used AWS CLI (`aws s3 sync --no-sign-request s3://physionet-open/sleep-edfx/1.0.0/sleep-cassette/ data/raw/`) to download all Sleep-Cassette files directly from PhysioNet. | Downloaded all raw `.edf` PSG signals and hypnograms (~8.1 GB) into local `data/raw/` folder. | ✅ Done | Test reading one patient recording with Python. |
| **2026-09-19** | **Tested Single Subject EEG Extraction** | Wrote `scripts/test_one_subject.py` using `mne` and `numpy` to inspect raw EDF signals, test extracting the `EEG Fpz-Cz` channel, filter between 0.5–35 Hz, and align annotations to 30-second epochs. | Successfully extracted subject `SC4001E0` with 0 signal clipping or timing drift. Verified 100 Hz sampling rate (3,000 samples per 30s epoch). | ✅ Done | Scale up preprocessing to all 39 recordings across 20 subjects. |
| **2026-09-19** | **Full Sleep-EDF Dataset Preprocessing** | Created `scripts/prepare_all_data.py`. Filtered 39 recordings for the Sleep-EDF-20 benchmark. Extracted `EEG Fpz-Cz`, bandpass filtered (0.5–35 Hz), segmented into 30s epochs, mapped stages to 5 AASM classes (Wake, N1, N2, N3, REM), and generated dataset index. | Generated **42,308 epochs** (~352.6 hours of clinical sleep). Saved processed `.npz` files in `data/processed/` and created `data/processed/manifest.json`. | ✅ Done | Create team worksheet system for Group A14. |
| **2026-09-19** | **Team Worksheets & Repository Setup** | Set up clean, conflict-free worksheets in `worksheets/` for all 3 members (Pankaj, Anurag, Sandesh), configured `.gitignore` to keep 8.1 GB data out of Git, and updated `README.md`. | Conflict-free collaboration system ready for all 3 members pushing from their laptops to `SandeshSalavi/NeuroStage`. | ✅ Done | Push branch `chore/team-workflow` to GitHub and start model training. |

---

## 📝 Key Findings & Preprocessing Summary

### 1. Dataset Preprocessing Numbers
- **Source:** PhysioNet Sleep-EDF Database Expanded (`sleep-cassette`).
- **Standard Benchmark:** Sleep-EDF-20 (20 healthy subjects, 39 recordings).
- **Extracted Channel:** Single-channel `EEG Fpz-Cz` at 100 Hz.
- **Epoch Duration:** 30 seconds (3,000 data points per epoch).
- **Total Processed Epochs:** 42,308 (~352.6 hours of sleep).

### 2. Sleep Class Breakdown
- **Wake (W):** 8,285 epochs (19.6%)
- **N1 (Light Sleep):** 2,804 epochs (6.6%) *(Significant class imbalance — will use weighted loss during training)*
- **N2 (Core Sleep):** 17,799 epochs (42.1%)
- **N3 (Deep Sleep):** 5,703 epochs (13.5%)
- **REM (Dreaming):** 7,717 epochs (18.2%)

### 3. Purpose of `manifest.json`
- Index file storing subject IDs, night IDs, and class counts so PyTorch can train without loading 8 GB into memory at once.
- Enables **20-fold subject-independent cross-validation** to prevent data leakage between nights of the same subject.
