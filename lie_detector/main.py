import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")
import argparse
import cv2
import time
import os
import wave
import pyaudio
import threading
import shutil
import speech_recognition as sr
import math
from agents.lie_detect_agent import LieDetectAgent

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    print("Warning: mediapipe not available. Landmark visualization disabled.")
    MEDIAPIPE_AVAILABLE = False

# MediaPipe landmark indices for each tracked AU region
# Based on the 468-point Face Mesh topology
AU_LANDMARK_REGIONS = {
    "AU04 Brow Lowerer":  [9, 55, 107, 66, 105, 63, 70, 285, 336, 296, 334, 293],   # inner brows
    "AU12 Lip Corner":    [61, 291, 185, 40, 39, 37, 267, 269, 270, 409],             # lip corners
    "AU15 Lip Depress":   [17, 314, 405, 321, 375, 291, 61, 146, 91, 181, 84, 17],   # lower lip
    "AU20 Lip Stretch":   [61, 291, 306, 292, 407, 320, 76, 77, 90, 180, 85, 16],    # outer lip
    "AU45 Blink":         [159, 145, 33, 133, 386, 374, 362, 263],                    # both eyes
}

# Global constants for real-time capture
CHUNK_DURATION = 5 # Process 5 seconds of data at a time
TEMP_DIR = os.path.join(os.getcwd(), "temp_capture")
FPS = 20.0 # Approximate framerate

# Face cascade for fallback bounding box when MediaPipe is unavailable
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Color palette for AU regions
AU_COLORS = [
    (255, 100, 100),   # AU04 - blue
    (100, 255, 100),   # AU12 - green
    (100, 100, 255),   # AU15 - red
    (255, 255, 100),   # AU20 - cyan
    (255, 100, 255),   # AU45 - magenta
]

os.makedirs(TEMP_DIR, exist_ok=True)

class RealtimePipeline:
    def __init__(self, agent):
        self.agent = agent
        self.latest_decision = "Initializing..."
        self.latest_prob = 0.5
        self.lock = threading.Lock()
        self.active_threads = []
        
    def record_audio_chunk(self, duration, output_path):
        """Records a chunk of audio from default microphone."""
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        rate = 44100
        p = pyaudio.PyAudio()

        try:
            stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
            frames = []
            
            for _ in range(0, int(rate / chunk * duration)):
                 data = stream.read(chunk)
                 frames.append(data)
                 
            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open(output_path, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()
        except Exception as e:
            print(f"Error recording audio: {e}")

    def capture_video_chunk(self, cap, duration, output_path):
        """Captures a chunk of video using an existing cv2 VideoCapture."""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(output_path, fourcc, FPS, (w, h))
        
        start_time = time.time()
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
            
        out.release()

    def transcribe_audio(self, audio_path):
        """Transcribes audio using SpeechRecognition."""
        if not os.path.exists(audio_path):
            print("[Audio] No audio file found - microphone may not be recording.")
            return ""
        
        file_size = os.path.getsize(audio_path)
        if file_size < 5000:
            print(f"[Audio] Audio file too small ({file_size} bytes) - mic may be off or silent.")
            return ""
        
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                print(f"[Audio] Transcript: '{text}'")
                return text
        except sr.UnknownValueError:
            print("[Audio] Could not understand speech (try speaking louder/clearer).")
            return ""
        except sr.RequestError as e:
            print(f"[Audio] Google API error (check internet): {e}")
            return ""
        except Exception as e:
            print(f"[Audio] Transcription error: {e}")
            return ""

    def process_chunk(self, video_path, audio_path):
        """Runs the agent on the captured chunks in a background thread."""
        chunk_name = os.path.basename(video_path)
        text = self.transcribe_audio(audio_path)
        
        print(f"\n[Started {chunk_name}] Transcribed Text: '{text}'")
        start_t = time.time()
        
        result = self.agent.analyze(image_or_video=video_path, audio_file=audio_path, text=text)
        
        with self.lock:
             self.latest_decision = result['decision']
             self.latest_prob = result['deception_prob']
             
        dt = time.time() - start_t
        v_score = result['scores'].get('vision', 0.5)
        a_score = result['scores'].get('audio', 0.5)
        print(f"\n[{chunk_name} Finished in {dt:.1f}s] Result: {self.latest_decision} (Vision: {v_score*100:.1f}%, Audio: {a_score*100:.1f}%)")

    def run(self):
         print("Starting live continuous capture. Press 'q' in the video window to quit.")
         
         # Open Webcam
         cap = cv2.VideoCapture(0)
         if not cap.isOpened():
             print("Error: Could not open webcam.")
             return
         
         # Initialize MediaPipe Face Mesh (persistent across frames for performance)
         face_mesh_detector = None
         if MEDIAPIPE_AVAILABLE:
             face_mesh_detector = mp_face_mesh.FaceMesh(
                 max_num_faces=1,
                 refine_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5
             )
             
         chunk_index = 0
         last_process_time = time.time()
         
         # To capture video for the model while displaying live
         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
         w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
         h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
         
         current_vid_path = os.path.join(TEMP_DIR, f"vid_{chunk_index}.mp4")
         current_out = cv2.VideoWriter(current_vid_path, fourcc, FPS, (w, h))
         
         # We also need to record continuous audio
         p = pyaudio.PyAudio()
         audio_format = pyaudio.paInt16
         audio_channels = 1
         audio_rate = 44100
         audio_chunk_size = 1024
         try:
             audio_stream = p.open(format=audio_format, channels=audio_channels, rate=audio_rate, 
                                   input=True, frames_per_buffer=audio_chunk_size)
         except Exception as e:
             print(f"Error opening microphone: {e}")
             return
             
         audio_frames = []

         try:
             while True:
                 ret, frame = cap.read()
                 if not ret: break
                 
                 # 1. Grab Audio frame
                 try:
                     audio_data = audio_stream.read(audio_chunk_size, exception_on_overflow=False)
                     audio_frames.append(audio_data)
                 except: pass

                 # 2. Add to current video chunk
                 current_out.write(frame)
                 
                 # 3. Draw face mesh and AU landmarks on display frame
                 display_frame = frame.copy()
                 h_frame, w_frame = display_frame.shape[:2]
                 
                 if MEDIAPIPE_AVAILABLE:
                     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     results = face_mesh_detector.process(rgb_frame)
                     if results.multi_face_landmarks:
                         for face_landmarks in results.multi_face_landmarks:
                             # Draw subtle full face mesh
                             mp_drawing.draw_landmarks(
                                 image=display_frame,
                                 landmark_list=face_landmarks,
                                 connections=mp_face_mesh.FACEMESH_TESSELATION,
                                 landmark_drawing_spec=None,
                                 connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                             )
                             # Highlight tracked AU regions with colored dots
                             lm = face_landmarks.landmark
                             for idx, (au_name, indices) in enumerate(AU_LANDMARK_REGIONS.items()):
                                 color = AU_COLORS[idx % len(AU_COLORS)]
                                 for li in indices:
                                     x_px = int(lm[li].x * w_frame)
                                     y_px = int(lm[li].y * h_frame)
                                     cv2.circle(display_frame, (x_px, y_px), 4, color, -1)
                             
                             # Draw AU legend in top-right corner
                             legend_x = w_frame - 220
                             legend_y = 20
                             cv2.rectangle(display_frame, (legend_x - 5, legend_y - 5),
                                         (w_frame - 5, legend_y + len(AU_LANDMARK_REGIONS) * 22 + 5),
                                         (30, 30, 30), -1)
                             for idx, au_name in enumerate(AU_LANDMARK_REGIONS.keys()):
                                 color = AU_COLORS[idx % len(AU_COLORS)]
                                 cv2.circle(display_frame, (legend_x + 8, legend_y + idx * 22 + 8), 6, color, -1)
                                 cv2.putText(display_frame, au_name, (legend_x + 20, legend_y + idx * 22 + 13),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
                 else:
                     # Fallback: simple bounding box
                     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                     for (x, y, w_box, h_box) in faces:
                         cv2.rectangle(display_frame, (x, y), (x+w_box, y+h_box), (255, 255, 255), 2)

                 # Draw probability bar and decision label at the bottom
                 with self.lock:
                      dec = self.latest_decision
                      prob = self.latest_prob

                 if prob is None or math.isnan(prob):
                     color = (200, 200, 200)
                     prob_disp = 0.5
                 else:
                     color = (0, int(255*(1-prob)), int(255*prob))
                     prob_disp = prob

                 # Bottom status bar
                 bar_h = 50
                 cv2.rectangle(display_frame, (0, h_frame - bar_h), (w_frame, h_frame), (30, 30, 30), -1)
                 bar_fill = int(w_frame * prob_disp)
                 cv2.rectangle(display_frame, (0, h_frame - bar_h), (bar_fill, h_frame), color, -1)
                 label = f"{dec}  |  Deception: {prob_disp*100:.1f}%"
                 cv2.putText(display_frame, label, (10, h_frame - 15),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                 cv2.imshow('Multi-Modal Lie Detector', display_frame)
                 
                 # 4. Check if it's time to process a chunk
                 if time.time() - last_process_time >= CHUNK_DURATION:
                     # Close current video writer
                     current_out.release()
                     
                     # Save current audio frames
                     current_aud_path = os.path.join(TEMP_DIR, f"aud_{chunk_index}.wav")
                     wf = wave.open(current_aud_path, 'wb')
                     wf.setnchannels(audio_channels)
                     wf.setsampwidth(p.get_sample_size(audio_format))
                     wf.setframerate(audio_rate)
                     wf.writeframes(b''.join(audio_frames))
                     wf.close()
                     
                     # Start a background thread to process the completed chunk
                     t = threading.Thread(target=self.process_chunk, args=(current_vid_path, current_aud_path))
                     t.start()
                     self.active_threads.append(t)
                     
                     # Clean up completed threads to avoid list growing forever
                     self.active_threads = [th for th in self.active_threads if th.is_alive()]
                     
                     # Setup for next chunk
                     chunk_index += 1
                     last_process_time = time.time()
                     current_vid_path = os.path.join(TEMP_DIR, f"vid_{chunk_index}.mp4")
                     current_out = cv2.VideoWriter(current_vid_path, fourcc, FPS, (w, h))
                     audio_frames = []

                 if cv2.waitKey(1) & 0xFF == ord('q'):
                     break
         finally:
             cap.release()
             cv2.destroyAllWindows()
             audio_stream.stop_stream()
             audio_stream.close()
             p.terminate()
             if hasattr(current_out, 'release'):
                 current_out.release()
             # Wait for background tasks to finish before cleaning up
             self.active_threads = [th for th in self.active_threads if th.is_alive()]
             if self.active_threads:
                 print(f"\nWaiting for {len(self.active_threads)} background processing tasks to finish (timeout 15s/task)...")
                 for t in self.active_threads:
                     t.join(timeout=15.0)
             # Clean up temp dir
             shutil.rmtree(TEMP_DIR, ignore_errors=True)
             print("\nExited gracefully.")

def main():
    parser = argparse.ArgumentParser(description="Multi-modal Lie Detection System")
   
    parser.add_argument("--openface-path", type=str, 
                    default=r"C:\Users\ayush\.gemini\antigravity\scratch\Final-Year-Project\lie_detector\agents\OpenFace\OpenFace_2.2.0_win_x64\FeatureExtraction.exe",
                    help="Path to OpenFace FeatureExtraction executable")
                        
    args = parser.parse_args()

    agent = LieDetectAgent(openface_path=args.openface_path)
    pipeline = RealtimePipeline(agent)
    pipeline.run()

if __name__ == "__main__":
    main()
