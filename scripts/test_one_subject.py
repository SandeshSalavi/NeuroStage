import os
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt

RAW_EVENT_ID = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}

# Mapping from raw MNE IDs to 5 AASM classes: W:0, N1:1, N2:2, N3:3, REM:4
AASM_MAP = {
    0: 0,  # Wake -> 0
    1: 1,  # N1 -> 1
    2: 2,  # N2 -> 2
    3: 3,  # N3 -> 3
    4: 3,  # N4 -> 3 (AASM N3)
    5: 4,  # REM -> 4
}

STAGE_NAMES = ["Wake (W)", "N1 (Light)", "N2 (Intermediate)", "N3 (Deep)", "REM"]

def process_subject(psg_path, hypno_path, out_path, channel="EEG Fpz-Cz"):
    print(f"\n[1/5] Loading Raw PSG: {psg_path}")
    raw = mne.io.read_raw_edf(psg_path, include=[channel], preload=True, verbose=False)
    
    print(f"      Original sampling rate: {raw.info['sfreq']} Hz | Duration: {raw.n_times / raw.info['sfreq'] / 3600:.2f} hrs")
    
    if raw.info["sfreq"] != 100.0:
        print("      Resampling to 100.0 Hz...")
        raw.resample(100.0)
    
    print("[2/5] Applying Bandpass Filter (0.3 - 35.0 Hz)...")
    raw.filter(l_freq=0.3, h_freq=35.0, verbose=False)
    
    print(f"[3/5] Attaching Hypnogram Annotations: {hypno_path}")
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
        tmax=30.0 - (1.0 / 100.0),  # Exactly 3000 points (0 to 29.99 s)
        baseline=None,
        preload=True,
        verbose=False
    )
    
    data = epochs.get_data(copy=True)  # Shape: (N, 1, 3000)
    raw_labels = epochs.events[:, 2]    # Raw class IDs (0 to 5)
    labels = np.array([AASM_MAP[lbl] for lbl in raw_labels], dtype=np.int64)
    
    print(f"      Total 30s epochs extracted: {len(labels)}")
    
    # Crop excessive Wake before first sleep and after last sleep (keep 30 min = 60 epochs)
    print("[4/5] Trimming pre-sleep and post-sleep Wake (AASM convention)...")
    sleep_idx = np.where(labels != 0)[0]
    if len(sleep_idx) > 0:
        first_sleep = max(0, sleep_idx[0] - 60)
        last_sleep = min(len(labels), sleep_idx[-1] + 61)
        data = data[first_sleep:last_sleep]
        labels = labels[first_sleep:last_sleep]
        print(f"      Epochs after wake trimming: {len(labels)} (Cropped from index {first_sleep} to {last_sleep})")
    
    # Per-subject Z-Score normalization
    print("[5/5] Normalizing signals (Subject Z-score)...")
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / (std + 1e-8)
    
    data = data.astype(np.float32)
    labels = labels.astype(np.int64)
    
    # Save output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, x=data, y=labels)
    print(f"\n[DONE] Saved processed file to: {out_path}")
    print(f"       Features shape: {data.shape} (N_epochs, Channels, Samples)")
    print(f"       Labels shape:   {labels.shape}")
    
    # Print Stage Breakdown
    unique, counts = np.unique(labels, return_counts=True)
    print("\nSleep Stage Breakdown:")
    for u, c in zip(unique, counts):
        pct = (c / len(labels)) * 100
        print(f"  Stage {u} [{STAGE_NAMES[u]:<19}]: {c:>5} epochs ({pct:>5.1f}%)")
    
    # Generate Visual Inspection Plots
    generate_plots(data, labels, out_path)

def generate_plots(data, labels, out_path):
    plot_prefix = out_path.replace(".npz", "")
    
    # 1. Hypnogram
    plt.figure(figsize=(14, 4))
    plt.step(range(len(labels)), labels, where='mid', color='#1d4ed8', lw=1.5)
    plt.yticks(range(5), STAGE_NAMES)
    plt.gca().invert_yaxis()  # Standard clinical hypnogram puts Wake at top, N3 at bottom
    plt.title("Clinical Ground Truth Hypnogram (Subject SC4001E0)", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch Index (30-second intervals)")
    plt.ylabel("Sleep Stage")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    hypno_img = f"{plot_prefix}_hypnogram.png"
    plt.savefig(hypno_img, dpi=150)
    plt.close()
    print(f"       Generated Plot: {hypno_img}")
    
    # 2. Sample 30s EEG Waveform (find an N2 epoch with spindles)
    n2_indices = np.where(labels == 2)[0]
    sample_idx = n2_indices[min(10, len(n2_indices)-1)] if len(n2_indices) > 0 else 50
    
    plt.figure(figsize=(14, 3))
    time_axis = np.linspace(0, 30, 3000)
    plt.plot(time_axis, data[sample_idx, 0], color='#047857', lw=0.8)
    plt.title(f"Preprocessed 30s EEG Waveform [Epoch #{sample_idx} - Stage: {STAGE_NAMES[labels[sample_idx]]}]", fontsize=11, fontweight='bold')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Amplitude (Z-Score)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    wave_img = f"{plot_prefix}_waveform.png"
    plt.savefig(wave_img, dpi=150)
    plt.close()
    print(f"       Generated Plot: {wave_img}")

if __name__ == "__main__":
    psg = "data/raw/SC4001E0-PSG.edf"
    hypno = "data/raw/SC4001EC-Hypnogram.edf"
    out = "data/processed/SC4001E0.npz"
    process_subject(psg, hypno, out)
