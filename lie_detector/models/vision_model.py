import os
import subprocess
import pandas as pd
import numpy as np

class VisionModel:
    def __init__(self, openface_path=None):
        """
        Initializes the OpenFace wrapper for Vision-based deception detection.
        
        Args:
            openface_path (str): The absolute path to FeatureExtraction.exe. 
                                 If None, assumes it is in the system PATH or the current directory.
        """
        # Default to checking if FeatureExtraction.exe exists in a common relative path, else just use the name
        self.openface_path = openface_path if openface_path else "FeatureExtraction.exe"
        self.output_dir = os.path.join(os.getcwd(), "openface_output")
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_features(self, video_path):
        """
        Runs OpenFace FeatureExtraction on the given video.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            str: Path to the generated CSV file, or None if extraction failed.
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found.")
            return None

        # Build OpenFace command
        # -f : input video
        # -out_dir : where to save outputs
        # -2Dfp, -3Dfp, -pdmparams, -pose, -aus, -gaze : which features to extract (we primarily want -aus and -pose)
        cmd = [
            self.openface_path,
            "-f", video_path,
            "-out_dir", self.output_dir,
            "-aus", # Extract Action Units
            "-pose" # Extract Head Pose
        ]

        try:
            # Run OpenFace synchronously
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # OpenFace names the output CSV the same as the input video file (without extension)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            csv_path = os.path.join(self.output_dir, f"{base_name}.csv")
            
            if os.path.exists(csv_path):
                return csv_path
            else:
                print("OpenFace did not generate the expected CSV file.")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"OpenFace execution failed: {e}")
            return None
        except FileNotFoundError:
             print(f"OpenFace executable not found at '{self.openface_path}'. Please verify the path.")
             return None

    def predict_deception(self, video_path):
        """
        Predicts deception probability based on extracted visual features (AUs).
        
        Args:
            video_path (str): Path to the video clip.
            
        Returns:
            float: A probability score where >0.5 leans towards deception.
        """
        csv_path = self.extract_features(video_path)
        if not csv_path:
            return 0.5 # Neutral fallback if feature extraction fails
            
        try:
            # Read the generated Action Units data
            df = pd.read_csv(csv_path)
            
            # Typical indicators of deception/stress in AUs (Simplified heuristic model)
            # AU04 (Brow Lowerer), AU15 (Lip Corner Depressor), AU20 (Lip Stretcher), 
            # AU12 (Lip Corner Puller - Fake smile), AU45 (Blink - increased blink rate)
            # Note: Columns have trailing spaces in OpenFace output, e.g., ' AU04_c'
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Target features for our heuristic stress model
            target_aus = ['AU04_c', 'AU15_c', 'AU45_c', 'AU12_c', 'AU20_c']
            
            stress_score = 0.0
            valid_frames = 0
            
            for index, row in df.iterrows():
                # Check if face was detected confidently in this frame
                if row.get('confidence', 0) > 0.7:
                    frame_stress = 0
                    # Sum presence (0 or 1) of stress-related AUs
                    for au in target_aus:
                        if au in df.columns:
                            frame_stress += row[au]
                            
                    stress_score += frame_stress
                    valid_frames += 1
            
            if valid_frames == 0:
                return 0.5 # No confident faces detected
                
            # Normalize score
            avg_stress_per_frame = stress_score / valid_frames
            
            # Map avg_stress (e.g. 0 to 5) to a probability sigmoid roughly
            # This is a placeholder heuristic weighting.
            # In a real system, this would be a trained classifier (SVM/NN) on these features.
            prob = 1 / (1 + np.exp(-1.5 * (avg_stress_per_frame - 0.5)))
            
            return float(prob)
            
        except Exception as e:
            print(f"Error processing Vision CSV: {e}")
            return 0.5
