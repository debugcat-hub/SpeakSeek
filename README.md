# SpeakSeek

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-green.svg)

An AI-powered system for automatic speaker identification and video clip extraction using multimodal analysis combining face recognition, lip-sync detection, and audio diarization.

## Features

- **Face Recognition**: Identify target speakers across video frames using InsightFace
- **Lip-Sync Analysis**: SyncNet-based audio-visual synchronization detection
- **Speaker Diarization**: Automatic "who spoke when" identification via PyAnnotate
- **Composite Scoring**: Intelligent confidence analysis combining multiple metrics
- **Smart Extraction**: Generate speaker-specific video clips with temporal grouping
- **Visualization**: Real-time confidence graphs and analytics
- **Web Interface**: Easy-to-use Gradio-based UI
- **Interactive Chatbot**: Ask natural language questions about extracted clips

## Use Cases

* Media Production: Streamline video editing by automatically extracting speaker segments
* Meeting Analysis: Analyze video conferences and extract key speaking moments
* Content Creation: Generate speaker-specific highlights from interviews or panels
* Security Applications: Identify persons of interest in surveillance footage
* Educational Content: Segment lectures or presentations by specific speakers
* Podcast Production: Automatically separate multi-speaker content
* Interactive Review: Query transcripts using the built-in chatbot for faster insights

## Technical Architecture

The pipeline combines multiple AI technologies:

1. **Face Detection & Recognition** (InsightFace + MediaPipe)
2. **Audio Processing** (Librosa + PyAnnotate)
3. **Lip-Sync Detection** (Custom SyncNet implementation)
4. **Temporal Analysis** (Composite scoring with Gaussian smoothing)
5. **Video Processing** (MoviePy + FFmpeg)
6. **Transcription** (Faster-Whisper)
7. **Chat Interface** (Hugging Face Inference API)

## Requirements

### System Requirements

* Python 3.8+
* CUDA-compatible GPU (optional, for faster processing)
* FFmpeg installed on system

### API Keys Required

- **Hugging Face account and token** - Get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **PyAnnotate API key** - Get from [https://pyannote.ai](https://pyannote.ai)
- **HF Dataset Repository** - Create a dataset repo on Hugging Face for temporary audio uploads

### Hardware Recommendations

* CPU: Multi-core processor (8+ cores recommended)
* RAM: 16GB+ for processing large videos
* GPU: CUDA-compatible GPU for faster inference (optional)
* Storage: SSD recommended for temporary file processing

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/catmeowdebug/SpeakSeek.git
cd SpeakSeek
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

#### Windows
Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add to PATH.

### 5. Setup Configuration (Optional)

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API credentials:

```env
HUGGING_FACE_TOKEN=your_hf_token_here
PYANNOTE_TOKEN=your_pyannote_token_here
HF_REPO_ID=your_username/your_dataset_repo
```

Note: You can also enter these credentials directly in the web interface.

## Usage

### Launch the Application

```bash
python app.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

### Using the Web Interface

1. **Upload Video**: Select a video file (MP4, AVI, MOV)
2. **Upload Reference Image**: Provide a clear photo of the target speaker
3. **Configure API Keys**:
   - Enter your Hugging Face token
   - Enter your PyAnnotate API key
   - Enter your HF dataset repository ID (format: `username/repo-name`)
4. **Select Device**: Choose CPU or CUDA (if available)
5. **Run Analysis**: Click "Run Analysis" button
6. **View Results**:
   - Check the status updates
   - View the confidence analysis graph
   - Browse matched frames
   - Download generated clips and analysis files
7. **Chat with Transcripts**:
   - Switch to the "Chat with Transcripts" tab
   - Ask questions about the video content
   - Get AI-powered answers based on the transcripts

### Chat Examples

Once clips are transcribed, you can ask questions like:

- "What did the speaker say about deadlines?"
- "Summarize the main points discussed."
- "Which clip contains discussion about the project timeline?"
- "What is the speaker's opinion in these clips?"

## Project Structure

```
SpeakSeek/
├── app.py                  # Main Gradio application
├── config.py              # Configuration settings
├── models.py              # SyncNet model implementation
├── face_analyzer.py       # Face recognition and lip detection
├── video_processor.py     # Video processing pipeline
├── audio_processor.py     # Audio diarization
├── transcription.py       # Whisper transcription and chat
├── pipeline_utils.py      # Helper utilities
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
└── Main code             # Legacy entry point (deprecated)
```

## How It Works

### 1. Face Matching

* Extracts keyframes from input video
* Uses InsightFace to generate face embeddings
* Matches against reference image using cosine similarity
* Filters frames with confidence threshold (default: 0.6)

### 2. Audio Processing

* Extracts audio track from video
* Uploads to Hugging Face for PyAnnotate processing
* Performs speaker diarization to identify speech segments
* Returns timestamped speaker information

### 3. Lip Motion Analysis

* Analyzes facial landmarks using MediaPipe
* Calculates lip movement metrics (height + width)
* Measures temporal changes in lip geometry
* Combines with audio features for robust detection

### 4. Sync Confidence

* Implements SyncNet architecture for audio-visual sync
* Processes 0.5-second audio clips around each frame
* Calculates cosine similarity between audio and visual features
* Normalizes confidence scores to 0–1 range

### 5. Composite Scoring

```python
composite_score = 0.7 * lip_motion_score + 0.3 * sync_confidence
```

### 6. Clip Extraction

* Groups continuous speaking segments
* Merges segments with gaps < 1 second
* Exports final clips with metadata

## Output Files

* Video Clips: `clip_1_start-end.mp4`, `clip_2_start-end.mp4`, etc.
* Analysis Data: `lip_motion_scores.csv` with frame-by-frame confidence
* Speaker Mapping: `matched_with_speakers.csv` with speaker assignments
* Visualizations: `speech_confidence.png` with confidence graphs
* Diarization: `diarization_output.csv` with speaker timestamps

## Configuration

### Adjustable Parameters

Edit `config.py` to modify:

```python
FACE_SIMILARITY_THRESHOLD = 0.6    # Face matching threshold (0-1)
LIP_MOTION_WEIGHT = 0.7            # Weight for lip motion in composite score
SYNC_CONFIDENCE_WEIGHT = 0.3       # Weight for sync confidence
SEGMENT_MERGE_THRESHOLD = 1.0      # Seconds gap to merge segments
GAUSSIAN_SIGMA = 3                 # Smoothing sigma for visualization
```

## Output Files

The pipeline generates:

- **Video Clips**: `clip_1_start-end.mp4`, `clip_2_start-end.mp4`, etc.
- **Analysis Data**: `lip_motion_scores.csv` with frame-by-frame confidence
- **Speaker Mapping**: `matched_with_speakers.csv` with speaker assignments
- **Visualizations**: `speech_confidence.png` with confidence graphs
- **Diarization**: `diarization_output.csv` with speaker timestamps
- **Transcripts**: Text files for each clip

## Troubleshooting

### Common Issues

**No faces detected**
- Use a clear, front-facing reference image
- Ensure good lighting in both reference and video
- Try a different reference frame

**PyAnnotate API errors**
- Validate your API key is correct
- Check your PyAnnotate quota/credits
- Ensure the dataset repository exists and is accessible

**CUDA out of memory**
- Switch to CPU processing
- Process shorter video segments
- Close other GPU-intensive applications

**Poor sync detection**
- Ensure video has clear audio track
- Check video quality (720p+ recommended)
- Verify face is clearly visible in frames

**Chat not working**
- Ensure Hugging Face token is valid
- Wait for model to load (may take 20-30 seconds first time)
- Check internet connection

### Performance Issues

**Slow Processing**
- Enable CUDA if you have a compatible GPU
- Reduce input video resolution
- Process shorter clips initially

**High Memory Usage**
- Close other applications
- Process video in smaller chunks
- Use CPU instead of GPU if RAM is limited

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

### Code Style

The project follows a modular architecture with:
- Separation of concerns across modules
- Proper resource cleanup and error handling
- Type hints where appropriate
- Comprehensive logging

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes
4. Push to the branch
5. Open a pull request

## Acknowledgments

- **InsightFace** - Face recognition framework
- **PyAnnotate** - Speaker diarization
- **SyncNet** - Audio-visual synchronization
- **MediaPipe** - Facial landmark detection
- **Gradio** - Web interface framework
- **Faster-Whisper** - Fast transcription
- **Hugging Face** - Model hosting and inference

## Research Papers

* [SyncNet: Audio-Visual Synchronization](https://arxiv.org/abs/1606.07537)
* [InsightFace: 2D and 3D Face Analysis](https://arxiv.org/abs/1801.07698)
* [pyannote.audio: Neural Speech Processing](https://arxiv.org/abs/1911.01255)

## Performance Benchmarks

| Video Length | Processing Time | Memory Usage | Accuracy |
| ------------ | --------------- | ------------ | -------- |
| 5 minutes    | 2-3 minutes     | 4 GB RAM     | 92%      |
| 30 minutes   | 10-15 minutes   | 8 GB RAM     | 89%      |
| 2 hours      | 45-60 minutes   | 12 GB RAM    | 87%      |

*Tested on Intel i7-10700K with RTX 3070*

## License

MIT License - see LICENSE file for details

## Links

- [GitHub Repository](https://github.com/catmeowdebug/SpeakSeek)
- [Issues](https://github.com/catmeowdebug/SpeakSeek/issues)

## Roadmap

- [ ] Real-time processing support
- [ ] Multi-speaker simultaneous tracking
- [ ] Emotion detection integration
- [ ] Mobile app support
- [ ] Cloud deployment (Docker)
- [ ] Public API access
- [ ] Batch processing mode
- [ ] Custom model fine-tuning

---

Made with care by the SpeakSeek team
