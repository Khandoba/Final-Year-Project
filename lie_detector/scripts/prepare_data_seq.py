import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vision_model import VisionModel
from models.audio_model import AudioModel
from moviepy import VideoFileClip

def normalize_sequence(seq, target_len=100):
    if len(seq) == 0:
        return np.zeros((target_len, seq.shape[1] if len(seq.shape)>1 else 1))
    if len(seq) == target_len:
        return seq
        
    old_indices = np.linspace(0, len(seq)-1, num=len(seq))
    new_indices = np.linspace(0, len(seq)-1, num=target_len)
    
    new_seq = np.zeros((target_len, seq.shape[1]))
    for i in range(seq.shape[1]):
        new_seq[:, i] = np.interp(new_indices, old_indices, seq[:, i])
    return new_seq

def prepare_dataset_seq():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'MU3D-Package'))
    videos_dir = os.path.join(data_dir, 'Videos', 'Videos')
    codebook_path = os.path.join(data_dir, 'MU3D Codebook.xlsx')
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    if not os.path.exists(codebook_path):
        print(f"Codebook not found at {codebook_path}")
        return

    print("Loading codebook...")
    codebook = pd.read_excel(codebook_path, sheet_name='Video-Level Data', engine='openpyxl')
    
    openface_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agents', 'OpenFace', 'OpenFace_2.2.0_win_x64', 'FeatureExtraction.exe'))
    vision_model = VisionModel(openface_path=openface_path)
    audio_model = AudioModel() # Ensure this is updated to use LLD first!
    
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.wmv')]
    target_aus = ['AU04_c', 'AU15_c', 'AU45_c', 'AU12_c', 'AU20_c']
    
    vision_seqs = []
    audio_seqs = []
    labels = []
    
    for i, video_filename in enumerate(video_files):
        print(f"Processing {i+1}/{len(video_files)}: {video_filename}")
        video_id = os.path.splitext(video_filename)[0]
        
        row = codebook[codebook['VideoID'] == video_id]
        if len(row) == 0:
            print(f"  Warning: {video_id} not found in codebook. Skipping.")
            continue
        label = row.iloc[0]['Veracity'] # 1=Truth, 0=Lie
        
        video_path = os.path.join(videos_dir, video_filename)
        
        # --- 1. Vision Features (Frame-by-frame) ---
        csv_path = vision_model.extract_features(video_path)
        v_seq = np.zeros((100, 5))
        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                # We want ALL frames to form a sequence, not just valid ones, so sequence represents time perfectly.
                # Just get the target AUs, filling NaNs with 0
                au_data = df[target_aus].fillna(0).values
                v_seq = normalize_sequence(au_data, target_len=100)
            except Exception as e:
                print(f"  Error parsing vision CSV: {e}")
        
        # --- 2. Audio Features (LLD) ---
        temp_audio_path = os.path.join(os.path.dirname(video_path), f"temp_{video_id}.wav")
        a_seq = np.zeros((100, 23)) # eGeMAPS LLD is 23 features
        try:
            with VideoFileClip(video_path) as clip:
                if clip.audio is not None:
                    clip.audio.write_audiofile(temp_audio_path, logger=None)
                    
            if os.path.exists(temp_audio_path):
                # We need to call extract_features which should now return sequence data
                feats = audio_model.extract_features(temp_audio_path)
                if feats is not None and len(feats) > 0:
                    a_seq = normalize_sequence(feats, target_len=100)
                os.remove(temp_audio_path)
        except Exception as e:
             print(f"  Audio extraction error for {video_id}: {e}")
             
        vision_seqs.append(v_seq)
        audio_seqs.append(a_seq)
        labels.append(label)
        
    print("Saving sequences...")
    np.save(os.path.join(out_dir, 'vision_seqs.npy'), np.array(vision_seqs))
    np.save(os.path.join(out_dir, 'audio_seqs.npy'), np.array(audio_seqs))
    np.save(os.path.join(out_dir, 'labels.npy'), np.array(labels))
    print("Done!")

if __name__ == "__main__":
    prepare_dataset_seq()
