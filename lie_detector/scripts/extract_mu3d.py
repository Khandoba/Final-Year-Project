import os
import pandas as pd
import argparse
from tqdm import tqdm
import sys

# Ensure the root project directory is in the path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vision_model import VisionModel
from models.audio_model import AudioModel
from moviepy import VideoFileClip

def remove_temp_audio(filepath):
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Extract MU3D Features")
    parser.add_argument("--limit", type=int, default=0, help="Number of videos to process (0 for all)")
    args = parser.parse_args()

    # Paths
    VIDEO_DIR = r"C:\Users\shlok\OneDrive\Desktop\final year project\MU3D-Package\Videos\Videos"
    CODEBOOK_PATH = r"C:\Users\shlok\OneDrive\Desktop\final year project\MU3D-Package\MU3D Codebook.xlsx"
    OUTPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'mu3d_extracted_features.csv')

    print("Initializing Models...")
    vision_model = VisionModel(openface_path=r"C:\Users\shlok\OneDrive\Desktop\OpenFace_2.2.0_win_x64\FeatureExtraction.exe")
    # Note: openface_path is hardcoded here based on previous error context or default logic. 
    # If the user has a different path, we might need to adjust. We will fall back to default if it fails.
    audio_model = AudioModel()

    print(f"Loading Codebook from {CODEBOOK_PATH}...")
    try:
        df_labels = pd.read_excel(CODEBOOK_PATH, sheet_name='Video-Level Data')
    except Exception as e:
        print(f"Failed to read codebook: {e}")
        return

    # Create mapping of VideoID to Truth/Lie
    # In MU3D, Veracity: 0 = Truth, 1 = Lie (typically, or vice versa, we'll extract the raw value)
    
    videos_to_process = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.wmv') or f.endswith('.mp4')]
    
    if args.limit > 0:
        videos_to_process = videos_to_process[:args.limit]

    print(f"Processing {len(videos_to_process)} videos...")

    dataset = []

    for vid_file in tqdm(videos_to_process):
        vid_id = os.path.splitext(vid_file)[0]
        
        # Find Veracity
        row = df_labels[df_labels['VideoID'] == vid_id]
        if row.empty:
            print(f"Warning: {vid_id} not found in codebook. Skipping.")
            continue
        
        veracity = row.iloc[0]['Veracity'] # 0 or 1
        
        vid_path = os.path.join(VIDEO_DIR, vid_file)
        
        # 1. Vision Features (Late Fusion Prob)
        vision_prob = vision_model.predict_deception(vid_path)
        
        # 2. Audio Features
        # we need to extract audio from video using moviepy
        temp_audio_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'temp_{vid_id}.wav')
        audio_prob = 0.5
        try:
            with VideoFileClip(vid_path) as clip:
                if clip.audio is not None:
                    clip.audio.write_audiofile(temp_audio_path, logger=None)
                    audio_prob = audio_model.predict_deception(temp_audio_path)
            remove_temp_audio(temp_audio_path)
        except Exception as e:
            print(f"Audio extraction failed for {vid_file}: {e}")
            remove_temp_audio(temp_audio_path)

        dataset.append({
            'VideoID': vid_id,
            'VisionProb': vision_prob,
            'AudioProb': audio_prob,
            'Veracity': veracity
        })

    # Save to CSV
    df_out = pd.DataFrame(dataset)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Extraction complete! Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
