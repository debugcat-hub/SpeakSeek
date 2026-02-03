#!/bin/bash

echo "==========================================="
echo "SpeakSeek Setup Script"
echo "==========================================="
echo

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    else
        echo "✓ $1 found"
        return 0
    fi
}

echo "Step 1: Checking system dependencies..."
check_command python3 || { echo "Please install Python 3.8+"; exit 1; }
check_command ffmpeg || { echo "Please install FFmpeg"; exit 1; }
echo

echo "Step 2: Checking Python version..."
python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" || {
    echo "Error: Python 3.8+ required"
    exit 1
}
echo "✓ Python version OK"
echo

echo "Step 3: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo

echo "Step 4: Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo

echo "Step 5: Upgrading pip..."
pip install --upgrade pip
echo

echo "Step 6: Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo

echo "Step 7: Setting up configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Created .env file from template"
    echo "Please edit .env and add your API credentials"
else
    echo "✓ .env file already exists"
fi
echo

echo "==========================================="
echo "Setup Complete!"
echo "==========================================="
echo
echo "Next steps:"
echo "1. Edit .env file and add your API credentials (or enter them in the UI)"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python app.py"
echo "4. Open http://localhost:7860 in your browser"
echo
