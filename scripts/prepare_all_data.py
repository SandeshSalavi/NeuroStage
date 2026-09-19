import os
import glob
import re
import json
import numpy as np
import mne
from tqdm import tqdm

RAW_EVENT_ID = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}

AASM_MAP = {
    0: 0,  # Wake -> 0
    1: 1,  # N1 -> 1
    2: 2,  # N2 -> 2
    3: 3,  # N3 -> 3
    4: 3,  # N4 -> 3 (Merged into AASM N3)
    5: 4,  # REM -> 4
}

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

def find_matched_pairs(raw_dir):
    """Find and pair all PSG files with their corresponding Hypnogram files."""
    psg_files = sorted(glob.glob(os.path.join(raw_dir, "*PSG.edf")))
    hypno_files = sorted(glob.glob(os.path.join(raw_dir, "*Hypnogram.edf")))
    
    pairs = []
    for psg in psg_files:
        filename = os.path.basename(psg)
        # Prefix is SC4 + subject(2 digits) + night(1 digit), e.g. SC4001, SC4002
        prefix = filename[:6]
        
        # Match with corresponding hypnogram
        matched_hypno = [h for h in hypno_files if os.path.basename(h).startswith(prefix)]
        if matched_hypno:
            subject_id = int(filename[3:5]) # e.g. SC4001 -> subject 0
            night_id = int(filename[5])     # e.g. SC4001 -> night 1
            pairs.append({
                "prefix": prefix,
                "subject_id": subject_id,
                "night_id": night_id,
                "psg": psg,
                "hypno": matched_hypno[0]
            })
    return pairs

def preprocess_recording(psg_path, hypno_path, channel="EEG Fpz-Cz"):
    # 1. Load EEG channel
    raw = mne.io.read_raw_edf(psg_path, include=[channel], preload=True, verbose=False)
    
    # 2. Resample to 100 Hz if needed
    if raw.info["sfreq"] != 100.0:
        raw.resample(100.0, verbose=False)
    
    # 3. Bandpass filter 0.3 - 35 Hz
    raw.filter(l_freq=0.3, h_freq=35.0, verbose=False)
    
    # 4. Read annotations
    annot = mne.read_annotations(hypno_path)
    raw.set_annotations(annot, emit_warning=False)
    
    events, event_id = mne.events_from_annotations(
        raw, 
        event_id=RAW_EVENT_ID, 
        chunk_duration=30.0, 
        verbose=False
    )
    
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=30.0 - (1.0 / 100.0), # 3000 samples
        baseline=None,
        preload=True,
        verbose=False
    )
    
    data = epochs.get_data(copy=True)
    raw_labels = epochs.events[:, 2]
    labels = np.array([AASM_MAP[lbl] for lbl in raw_labels], dtype=np.int64)
    
    # 5. Crop Wake to 30 mins before sleep onset and 30 mins after final awakening
    sleep_idx = np.where(labels != 0)[0]
    if len(sleep_idx) > 0:
        first_sleep = max(0, sleep_idx[0] - 60)
        last_sleep = min(len(labels), sleep_idx[-1] + 61)
        data = data[first_sleep:last_sleep]
        labels = labels[first_sleep:last_sleep]
    
    # 6. Per-subject Z-Score normalization
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / (std + 1e-8)
    
    return data.astype(np.float32), labels.astype(np.int64)

def run_pipeline(raw_dir="data/raw", out_dir="data/processed", max_subjects=None):
    os.makedirs(out_dir, exist_ok=True)
    pairs = find_matched_pairs(raw_dir)
    print(f"Total matched PSG-Hypnogram recordings found: {len(pairs)}")
    
    # If max_subjects specified (e.g. 20 for Sleep-EDF-20 benchmark)
    if max_subjects is not None:
        pairs = [p for p in pairs if p["subject_id"] < max_subjects]
        print(f"Filtered for Sleep-EDF-{max_subjects} benchmark: {len(pairs)} recordings across {max_subjects} subjects.")
    
    manifest = []
    total_epochs = 0
    total_class_counts = {name: 0 for name in STAGE_NAMES}
    
    for item in tqdm(pairs, desc="Preprocessing Sleep-EDF"):
        out_filename = f"{item['prefix']}.npz"
        out_path = os.path.join(out_dir, out_filename)
        
        # Skip if already preprocessed
        if not os.path.exists(out_path):
            try:
                x, y = preprocess_recording(item["psg"], item["hypno"])
                np.savez_compressed(out_path, x=x, y=y)
            except Exception as e:
                print(f"Error processing {item['prefix']}: {e}")
                continue
        else:
            saved = np.load(out_path)
            y = saved["y"]
        
        # Track statistics
        unique, counts = np.unique(y, return_counts=True)
        rec_counts = {STAGE_NAMES[u]: int(c) for u, c in zip(unique, counts)}
        for stage, c in rec_counts.items():
            total_class_counts[stage] += c
        
        total_epochs += len(y)
        manifest.append({
            "prefix": item["prefix"],
            "subject_id": item["subject_id"],
            "night_id": item["night_id"],
            "file": out_filename,
            "n_epochs": len(y),
            "class_distribution": rec_counts
        })
    
    # Save manifest.json
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "total_recordings": len(manifest),
            "total_epochs": total_epochs,
            "class_totals": total_class_counts,
            "recordings": manifest
        }, f, indent=2)
    
    print("\n" + "="*50)
    print(" PREPROCESSING COMPLETE")
    print("="*50)
    print(f"Total Processed Recordings: {len(manifest)}")
    print(f"Total 30s Epochs:          {total_epochs} (~{total_epochs*30/3600:.1f} hours of sleep)")
    print("Overall Class Distribution:")
    for stage, count in total_class_counts.items():
        pct = (count / total_epochs) * 100 if total_epochs > 0 else 0
        print(f"  {stage:<15}: {count:>6} epochs ({pct:>5.1f}%)")
    print(f"Manifest written to:        {manifest_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess Sleep-EDF dataset into clean .npz tensors")
    parser.add_argument("--raw_dir", default="data/raw", help="Path to raw EDF directory")
    parser.add_argument("--out_dir", default="data/processed", help="Path to save processed .npz files")
    parser.add_argument("--max_subjects", type=int, default=20, help="Number of subjects (e.g. 20 for Sleep-EDF-20 benchmark, or omit for all)")
    args = parser.parse_args()
    
    run_pipeline(args.raw_dir, args.out_dir, args.max_subjects)
