# Main - AI Voice Synthesis and Authentication Portal

Secure Access. Zero-Shot Voice Cloning. Seamless Experience.

Main is the secure authentication gateway and advanced neural voice cloning application built on top of the VoxClone architecture. It integrates a premium glassmorphism user interface with robust user management and high-fidelity text-to-speech features.

---

## Key Features

Secure Authentication Portal: Built-in Registration and Login system with encrypted session management.

Zero-Shot Voice Cloning: Clone any voice from a single short audio sample using F5-TTS.

High-Fidelity Synthesis: Powered by Faster-Whisper (Large-v3) and F5-TTS for natural and accurate audio generation.

Multi-Language Support: Full support for multiple languages including English, Urdu, Hindi, and more with Smart Text AutoFix.

Premium UI/UX: Stunning glassmorphism design, responsive layouts, and neural background animations.

GPU Accelerated: Optimized for Kaggle T4/P100 GPUs and Local environments with CUDA acceleration.

---

## Technology Stack

Frontend: HTML5, CSS3 (Glassmorphism, Animations), Vanilla JS

Backend: FastAPI, Uvicorn, Python 3.10+

AI Models: F5-TTS (Voice Synthesis), Faster-Whisper Large-v3 (STT)

Audio Processing: FFmpeg, SoundFile, NumPy

Infrastructure: Kaggle GPUs, Cloudflare Tunnel, LocalTunnel

---

## Quick Start

Prerequisites: Python 3.10+, CUDA-compatible GPU (Highly Recommended), FFmpeg installed on your system.

### Installation and Execution

Step 1: Clone the repository:
```bash
git clone https://github.com/Noorayhh/Vox-Clone-Voice-Cloning-App-.git
cd Vox-Clone-Voice-Cloning-App-
```

Step 2: Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
venv\Scripts\activate
```

Step 3: Install dependencies:
```bash
pip install fastapi uvicorn python-multipart pydub soundfile huggingface_hub faster-whisper vocos autocorrect nest-asyncio tqdm librosa
pip install git+https://github.com/SWivid/F5-TTS.git
npm install -g localtunnel
```

Step 4: Run the Application:
```bash
python voxclone_login.py
```

The server will start at http://localhost:8000. Create an account or log in to access the synthesis studio.

---

## Running on Kaggle

Step 1: Create a new Kaggle Notebook and enable GPU (T4 x2 or P100).

Step 2: Upload voxclone_login.py to the Kaggle working directory.

Step 3: Run the script. It will automatically bootstrap, install missing dependencies, and launch a public URL via LocalTunnel/Cloudflare.

Step 4: Open the public link, create an account, and start cloning!

---

## Author

Noor Fatima

GitHub: https://github.com/Noorayhh

Portfolio: https://noorayhh.github.io

Built with love and Neural Networks
