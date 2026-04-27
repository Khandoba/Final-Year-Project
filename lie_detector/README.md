# Multi-Modal Lie Detection System (Real-Time)

A real-time deception detection system built using an Agentic ReAct framework. This system analyzes visual, acoustic, and linguistic properties of a speaker in real-time to assess deception probabilities.

## Modalities
1. **Vision**: Uses [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) to extract Action Units (AUs) and head pose.
2. **Audio**: Uses [openSMILE](https://audeering.github.io/opensmile-python/) to extract eGeMAPSv02 acoustic features indicating vocal stress.
3. **Text**: Uses speech-to-text combined with a robust HuggingFace BERT model to analyze linguistic cues.

## Requirements

### Python Dependencies
Python >= 3.9
```bash
py -m pip install torch torchvision transformers opencv-python pandas numpy SpeechRecognition opensmile
```
*(If you do not have the `py` launcher, you must disable the Windows Python alias in Settings > Apps > Advanced app settings > App execution aliases)*
*(Or use `poetry install` with the provided `pyproject.toml`)*

### OpenFace Dependency
To utilize the Vision model, you must download the pre-compiled OpenFace binaries for Windows:
1. Download from [OpenFace Releases](https://github.com/TadasBaltrusaitis/OpenFace/releases).
2. Extract the folder.
3. Locate `FeatureExtraction.exe`.

## Running the Real-Time System

The main script runs an OpenCV webcam capture loop. It streams your webcam feed, draws a bounding box on your face, and continuously updates the Truth/Lie probability by analyzing 5-second sliding windows of video and audio in the background.

```bash
python main.py --openface-path "C:\path\to\OpenFace\FeatureExtraction.exe"
```

Press `q` on the OpenCV window to exit.

## Architecture

- `main.py`: Real-time capture loop, threading, and GUI.
- `agents/lie_detect_agent.py`: Agentic ReAct reasoning engine resolving conflicts between modalities.
- `models/`: Wrappers for Vision (OpenFace), Audio (openSMILE), and Text (BERT).
