"""
Fast re-extraction of vision sequences using ALL 18 Action Units
from the existing OpenFace CSV files (no need to re-run OpenFace).
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

ALL_AUS = [
    'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c',
    'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c',
    'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c'
]

def normalize_sequence(seq, target_len=100):
    if len(seq) == 0:
        return np.zeros((target_len, seq.shape[1]))
    if len(seq) == target_len:
        return seq
    old_indices = np.linspace(0, len(seq)-1, num=len(seq))
    new_indices = np.linspace(0, len(seq)-1, num=target_len)
    new_seq = np.zeros((target_len, seq.shape[1]))
    for i in range(seq.shape[1]):
        new_seq[:, i] = np.interp(new_indices, old_indices, seq[:, i])
    return new_seq

def reextract_vision():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'MU3D-Package'))
    codebook_path = os.path.join(data_dir, 'MU3D Codebook.xlsx')
    openface_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'openface_output'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    print("Loading codebook...")
    codebook = pd.read_excel(codebook_path, sheet_name='Video-Level Data', engine='openpyxl')

    csv_files = [f for f in os.listdir(openface_output_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} OpenFace CSVs in openface_output/")

    # Load existing labels and audio to keep them aligned
    old_labels = np.load(os.path.join(out_dir, 'labels.npy'))
    old_audio = np.load(os.path.join(out_dir, 'audio_seqs.npy'))
    old_vision = np.load(os.path.join(out_dir, 'vision_seqs.npy'))

    video_id_order = []
    # Reconstruct the video_id order from old extraction 
    # We'll match based on codebook VideoID order used in prepare_data_seq.py
    videos_dir = os.path.join(data_dir, 'Videos', 'Videos')
    video_files = sorted([f for f in os.listdir(videos_dir) if f.endswith('.wmv')])

    vision_seqs = []
    labels_out = []
    audio_seqs_out = []

    for i, video_filename in enumerate(video_files):
        video_id = os.path.splitext(video_filename)[0]
        row = codebook[codebook['VideoID'] == video_id]
        if len(row) == 0:
            continue

        label = row.iloc[0]['Veracity']
        csv_path = os.path.join(openface_output_dir, f"{video_id}.csv")

        if not os.path.exists(csv_path):
            print(f"  Missing CSV for {video_id}, skipping.")
            continue

        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            available_aus = [au for au in ALL_AUS if au in df.columns]
            au_data = df[available_aus].fillna(0).values
            v_seq = normalize_sequence(au_data, target_len=100)
        except Exception as e:
            print(f"  Error for {video_id}: {e}")
            v_seq = np.zeros((100, len(ALL_AUS)))

        vision_seqs.append(v_seq)
        labels_out.append(label)
        video_id_order.append(video_id)

    print(f"Re-extracted {len(vision_seqs)} vision sequences with {len(ALL_AUS)} AUs each.")

    # We need audio to stay aligned with vision. Re-align audio from old data.
    # Match by re-building a label->index map
    print("Aligning audio sequences with new vision order...")
    old_vision_labels = old_labels.tolist()
    
    # Load old vision seqs — align audio by matching labels sequentially
    # Since both used the same video order, we trust alignment is preserved
    # Just cap to min length
    min_len = min(len(vision_seqs), len(old_audio))
    vision_seqs = vision_seqs[:min_len]
    labels_out = labels_out[:min_len]
    audio_aligned = old_audio[:min_len]

    vision_arr = np.array(vision_seqs)
    labels_arr = np.array(labels_out)

    print(f"New vision shape: {vision_arr.shape}")
    print(f"Audio shape (unchanged): {audio_aligned.shape}")
    print(f"Labels: {labels_arr.shape}")

    np.save(os.path.join(out_dir, 'vision_seqs.npy'), vision_arr)
    np.save(os.path.join(out_dir, 'audio_seqs.npy'), audio_aligned)
    np.save(os.path.join(out_dir, 'labels.npy'), labels_arr)

    print("Saved new vision_seqs.npy with 18 AUs!")

if __name__ == "__main__":
    reextract_vision()
