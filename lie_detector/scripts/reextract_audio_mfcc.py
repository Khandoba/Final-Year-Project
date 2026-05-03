"""
Re-extract audio features using librosa 40-coefficient MFCCs.
Much richer than openSMILE LLDs for deception detection.
Extracts audio directly from the .wmv files using moviepy.
~20 minutes to process all 316 videos.
"""
import os
import sys
import numpy as np
import pandas as pd
import librosa

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from moviepy import VideoFileClip

N_MFCC = 40
TARGET_LEN = 100

def normalize_sequence(seq, target_len=TARGET_LEN):
    """Interpolate/resample a 2D sequence to target_len time steps."""
    if len(seq) == 0:
        return np.zeros((target_len, seq.shape[1] if len(seq.shape) > 1 else N_MFCC))
    if len(seq) == target_len:
        return seq
    old_idx = np.linspace(0, len(seq) - 1, num=len(seq))
    new_idx = np.linspace(0, len(seq) - 1, num=target_len)
    new_seq = np.zeros((target_len, seq.shape[1]))
    for i in range(seq.shape[1]):
        new_seq[:, i] = np.interp(new_idx, old_idx, seq[:, i])
    return new_seq

def extract_mfcc(audio_path, n_mfcc=N_MFCC):
    """Extract MFCC sequences from a WAV file using librosa."""
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=512)
        # mfcc shape: (n_mfcc, n_frames) -> transpose to (n_frames, n_mfcc)
        return mfcc.T
    except Exception as e:
        print(f"  librosa error: {e}")
        return None

def reextract_audio_mfcc():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'MU3D-Package'))
    videos_dir = os.path.join(data_dir, 'Videos', 'Videos')
    codebook_path = os.path.join(data_dir, 'MU3D Codebook.xlsx')
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    temp_dir = os.path.join(out_dir, 'temp_audio')
    os.makedirs(temp_dir, exist_ok=True)

    print("Loading codebook...")
    codebook = pd.read_excel(codebook_path, sheet_name='Video-Level Data', engine='openpyxl')

    # Load existing vision and labels to stay aligned
    vision_seqs = np.load(os.path.join(out_dir, 'vision_seqs.npy'))
    labels = np.load(os.path.join(out_dir, 'labels.npy'))

    video_files = sorted([f for f in os.listdir(videos_dir) if f.endswith('.wmv')])

    audio_seqs = []
    valid_count = 0

    for i, video_filename in enumerate(video_files):
        video_id = os.path.splitext(video_filename)[0]
        row = codebook[codebook['VideoID'] == video_id]
        if len(row) == 0:
            continue

        print(f"Processing {valid_count+1}/{len(labels)}: {video_filename}")
        video_path = os.path.join(videos_dir, video_filename)
        temp_wav = os.path.join(temp_dir, f"{video_id}.wav")

        a_seq = np.zeros((TARGET_LEN, N_MFCC))
        try:
            with VideoFileClip(video_path) as clip:
                if clip.audio is not None:
                    clip.audio.write_audiofile(temp_wav, logger=None)

            if os.path.exists(temp_wav):
                mfcc_seq = extract_mfcc(temp_wav)
                if mfcc_seq is not None and len(mfcc_seq) > 0:
                    a_seq = normalize_sequence(mfcc_seq, TARGET_LEN)
                os.remove(temp_wav)
        except Exception as e:
            print(f"  Error for {video_id}: {e}")

        audio_seqs.append(a_seq)
        valid_count += 1
        if valid_count >= len(labels):
            break

    audio_arr = np.array(audio_seqs)
    print(f"\nNew audio shape: {audio_arr.shape}")

    np.save(os.path.join(out_dir, 'audio_seqs.npy'), audio_arr)
    print("Saved audio_seqs.npy with 40-coefficient MFCCs!")

    # Cleanup temp dir
    try:
        os.rmdir(temp_dir)
    except:
        pass

if __name__ == "__main__":
    reextract_audio_mfcc()
