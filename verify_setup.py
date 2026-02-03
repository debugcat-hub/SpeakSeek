#!/usr/bin/env python3
"""
Verification script to check if SpeakSeek is properly set up.
Run this after installation to verify all dependencies are available.
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} - {str(e)}")
        return False

def check_system_command(command):
    """Check if a system command is available"""
    import shutil
    if shutil.which(command):
        print(f"✓ {command}")
        return True
    else:
        print(f"✗ {command} not found")
        return False

def main():
    print("=" * 50)
    print("SpeakSeek Setup Verification")
    print("=" * 50)
    print()

    all_ok = True

    print("Checking Python version...")
    version_info = sys.version_info
    if version_info >= (3, 8):
        print(f"✓ Python {version_info.major}.{version_info.minor}.{version_info.micro}")
    else:
        print(f"✗ Python {version_info.major}.{version_info.minor} (3.8+ required)")
        all_ok = False
    print()

    print("Checking system dependencies...")
    all_ok &= check_system_command("ffmpeg")
    print()

    print("Checking Python packages...")
    packages = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("gradio", "gradio"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("matplotlib", "matplotlib"),
        ("moviepy", "moviepy"),
        ("pydub", "pydub"),
        ("huggingface_hub", "huggingface-hub"),
        ("insightface", "insightface"),
        ("mediapipe", "mediapipe"),
        ("tqdm", "tqdm"),
        ("scipy", "scipy"),
        ("faster_whisper", "faster-whisper"),
        ("requests", "requests"),
    ]

    for module, package in packages:
        result = check_import(module, package)
        all_ok &= result

    print()

    print("Checking project modules...")
    project_modules = [
        "config",
        "models",
        "face_analyzer",
        "video_processor",
        "audio_processor",
        "transcription",
        "pipeline_utils",
        "app"
    ]

    for module in project_modules:
        result = check_import(module)
        all_ok &= result

    print()
    print("=" * 50)

    if all_ok:
        print("✓ All checks passed!")
        print()
        print("You're ready to run SpeakSeek:")
        print("  python app.py")
        print()
        print("Then open http://localhost:7860 in your browser")
    else:
        print("✗ Some checks failed")
        print()
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print()
        print("For system dependencies, check the README.md")

    print("=" * 50)
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
