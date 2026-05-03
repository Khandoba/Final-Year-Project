import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vision_model import VisionModel
from models.audio_model import AudioModel
from moviepy import VideoFileClip

def prepare_dataset(num_videos=None):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'MU3D-Package'))
    videos_dir = os.path.join(data_dir, 'Videos', 'Videos')
    codebook_path = os.path.join(data_dir, 'MU3D Codebook.xlsx')
    output_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data.csv'))

    if not os.path.exists(codebook_path):
        print(f"Codebook not found at {codebook_path}")
        return

    print("Loading codebook...")
    codebook = pd.read_excel(codebook_path, sheet_name='Video-Level Data', engine='openpyxl')
    
    # Initialize models for feature extraction
    # We pass the absolute path to OpenFace executable
    openface_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agents', 'OpenFace', 'OpenFace_2.2.0_win_x64', 'FeatureExtraction.exe'))
    vision_model = VisionModel(openface_path=openface_path)
    audio_model = AudioModel()
    
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.wmv')]
    if num_videos is not None:
        video_files = video_files[:num_videos]
    
    target_aus = ['AU04_c', 'AU15_c', 'AU45_c', 'AU12_c', 'AU20_c']
    
    dataset = []
    
    for i, video_filename in enumerate(video_files):
        print(f"Processing {i+1}/{num_videos}: {video_filename}")
        video_id = os.path.splitext(video_filename)[0]
        
        # Get label from codebook
        row = codebook[codebook['VideoID'] == video_id]
        if len(row) == 0:
            print(f"  Warning: {video_id} not found in codebook. Skipping.")
            continue
        label = row.iloc[0]['Veracity'] # 1=Truth, 0=Lie
        
        video_path = os.path.join(videos_dir, video_filename)
        
        # --- 1. Vision Features ---
        csv_path = vision_model.extract_features(video_path)
        vision_feats = [0.0] * 5
        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                valid_frames = df[df['confidence'] > 0.7]
                if len(valid_frames) > 0:
                    for j, au in enumerate(target_aus):
                        if au in valid_frames.columns:
                            vision_feats[j] = valid_frames[au].mean()
            except Exception as e:
                print(f"  Error parsing vision CSV: {e}")
        else:
            print(f"  Vision extraction failed for {video_id}")
            
        # --- 2. Audio Features ---
        temp_audio_path = os.path.join(os.path.dirname(video_path), f"temp_{video_id}.wav")
        audio_feats = np.zeros(88) # openSMILE eGeMAPSv02 has 88 features
        try:
            with VideoFileClip(video_path) as clip:
                if clip.audio is not None:
                    clip.audio.write_audiofile(temp_audio_path, logger=None)
                    
            if os.path.exists(temp_audio_path):
                feats = audio_model.extract_features(temp_audio_path)
                if feats is not None:
                    audio_feats = feats
                os.remove(temp_audio_path)
            else:
                print(f"  Audio extraction failed (no file) for {video_id}")
        except Exception as e:
             print(f"  Audio extraction error for {video_id}: {e}")
             
        # Create dataset row
        row_data = {'VideoID': video_id, 'Label': label}
        for j, au in enumerate(target_aus):
            row_data[au] = vision_feats[j]
        for j in range(88):
            row_data[f'audio_{j}'] = audio_feats[j]
            
        dataset.append(row_data)
        
    df_out = pd.DataFrame(dataset)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved {len(dataset)} samples to {output_csv}")

if __name__ == "__main__":
    prepare_dataset(None)
