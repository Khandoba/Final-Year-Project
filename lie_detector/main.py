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

# Global constants for real-time capture
CHUNK_DURATION = 5 # Process 5 seconds of data at a time
TEMP_DIR = os.path.join(os.getcwd(), "temp_capture")
FPS = 20.0 # Approximate framerate

# We need a quick face detector for drawing bounding boxes on the live feed
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                return text
        except sr.UnknownValueError:
            return "" # Could not understand audio
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""
        except Exception as e:
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
             self.latest_prob = result.get('fused_probability', 0.5)
             
        dt = time.time() - start_t
        print(f"\n[{chunk_name} Finished in {dt:.1f}s] Result: {self.latest_decision}")

    def run(self):
         print("Starting live continuous capture. Press 'q' in the video window to quit.")
         
         # Open Webcam
         cap = cv2.VideoCapture(0)
         if not cap.isOpened():
             print("Error: Could not open webcam.")
             return
             
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
                 
                 # 3. Draw bounding box and live stats on display frame
                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                 faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                 
                 with self.lock:
                      dec = self.latest_decision
                      prob = self.latest_prob
                      
                 # Determine color based on probability
                 # Green for Truth (low prob), Red for Lie (high prob)
                 if prob is None or math.isnan(prob):
                     color = (255, 255, 255)
                     prob_disp = 0.5
                 else:
                     color = (0, int(255*(1-prob)), int(255*prob))
                     prob_disp = prob
                 
                 for (x, y, w_box, h_box) in faces:
                     cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 2)
                     text_label = f"{dec}: {prob_disp*100:.1f}%"
                     cv2.putText(frame, text_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                     
                 cv2.imshow('Multi-Modal Lie Detector', frame)
                 
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
    parser.add_argument("--openface-path", type=str, default="FeatureExtraction.exe",
                        help="Path to OpenFace FeatureExtraction executable")
    args = parser.parse_args()

    agent = LieDetectAgent(openface_path=args.openface_path)
    pipeline = RealtimePipeline(agent)
    pipeline.run()

if __name__ == "__main__":
    main()
