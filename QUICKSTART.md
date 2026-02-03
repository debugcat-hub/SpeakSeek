# Quick Start Guide

Get SpeakSeek running in 5 minutes!

## Prerequisites

Before you start, make sure you have:

1. Python 3.8 or higher installed
2. FFmpeg installed on your system
3. A Hugging Face account and API token
4. A PyAnnotate API key

## Installation

### Automated Setup (Linux/Mac)

```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create config file (optional)
cp .env.example .env
```

## Configuration

You can configure API credentials in two ways:

### Option 1: Environment File (Optional)

Edit `.env`:
```
HUGGING_FACE_TOKEN=hf_your_token_here
PYANNOTE_TOKEN=your_pyannote_key_here
HF_REPO_ID=username/dataset-repo
```

### Option 2: Web Interface (Recommended)

Enter credentials directly in the web interface when you run the app.

## Running the Application

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the application
python app.py
```

Open your browser to [http://localhost:7860](http://localhost:7860)

## First Time Usage

1. Upload a video file (MP4, AVI, MOV)
2. Upload a clear reference photo of the target speaker
3. Enter your API credentials:
   - Hugging Face token
   - PyAnnotate API key
   - HF dataset repository ID (e.g., `username/my-dataset`)
4. Click "Run Analysis"
5. Wait for processing to complete (2-10 minutes depending on video length)
6. View results and download clips
7. Switch to "Chat with Transcripts" tab to ask questions

## Example Questions for Chat

Once processing is complete, try asking:

- "What did the speaker say about deadlines?"
- "Summarize the main points discussed"
- "Which clip talks about the project timeline?"
- "What is the speaker's opinion on this topic?"

## Troubleshooting

### "No faces detected"
Use a clear, well-lit reference photo showing the face straight on.

### "PyAnnotate API error"
Verify your API key at [https://pyannote.ai](https://pyannote.ai)

### "Out of memory"
- Switch to CPU processing
- Try a shorter video first
- Close other applications

### Chat not responding
- Verify your Hugging Face token
- Wait 20-30 seconds for the model to load
- Check your internet connection

## Getting API Keys

### Hugging Face Token
1. Sign up at [https://huggingface.co](https://huggingface.co)
2. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Create a new token with "read" permissions
4. Copy the token (starts with `hf_`)

### PyAnnotate API Key
1. Sign up at [https://pyannote.ai](https://pyannote.ai)
2. Navigate to API section
3. Copy your API key

### HF Dataset Repository
1. Go to [https://huggingface.co/new-dataset](https://huggingface.co/new-dataset)
2. Create a new dataset repository
3. Use format: `your-username/dataset-name`
4. The repository is used for temporary audio uploads

## Next Steps

- Check out the full [README.md](README.md) for detailed documentation
- Adjust configuration in `config.py` to fine-tune performance
- Report issues at [GitHub Issues](https://github.com/catmeowdebug/SpeakSeek/issues)

---

Happy analyzing!
