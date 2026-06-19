import os, sys, subprocess, time, threading, re, base64, shutil, gc, asyncio

# --- 1. STABLE & CLEAN BOOTSTRAPPER ---
def setup():
    if not os.path.exists("/kaggle"): return 
    base = "/kaggle/working"
    for f in os.listdir(base):
        if f.endswith(".wav") or f.startswith("clone_") or f.startswith("morph_") or f.startswith("r_") or f.startswith("o_"):
            try: os.remove(os.path.join(base, f))
            except: pass
    os.environ["HF_HOME"] = base
    os.environ["XDG_CACHE_HOME"] = base
    os.environ["TRANSFORMERS_CACHE"] = base
    os.environ["HF_HUB_CACHE"] = base
    try: 
        subprocess.run(["fuser", "-k", "8000/tcp"], capture_output=True)
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except: pass
    
    try:
        import f5_tts, faster_whisper, noisereduce, nest_asyncio
    except ImportError:
        print("\n[SYSTEM] DOWNLOADING DEPENDENCIES (NO RESTART LOOP)... PLEASE WAIT.\n")
        libs = ["fastapi", "uvicorn", "python-multipart", "pydub", "soundfile", 
                "huggingface_hub", "faster-whisper", "vocos", "autocorrect", "nest-asyncio", "tqdm", "librosa"]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + libs, check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/SWivid/F5-TTS.git"], check=False)
        subprocess.run(["npm", "install", "-g", "localtunnel"], capture_output=True) # Add LocalTunnel
        import importlib
        importlib.invalidate_caches()
        print("\n[SYSTEM] DOWNLOAD COMPLETE. STARTING SERVER IN ONE GO...\n")

setup()

import torch, uvicorn, nest_asyncio, nltk, numpy as np, soundfile as sf

# --- 2. MASTER ENGINE ---
from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import snapshot_download
import nest_asyncio

nest_asyncio.apply()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

debug_logs = []
def log_debug(msg):
    timestamp = time.strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    debug_logs.append(full_msg)
    if len(debug_logs) > 50:
        debug_logs.pop(0)

@app.get("/api/debug")
async def get_debug_logs():
    return {"logs": debug_logs}

is_kag = os.path.exists("/kaggle")
v_tmp = "/kaggle/working" if is_kag else "."

def is_ref_audio_prepended(wave, ref_file, target_sr=24000):
    try:
        import soundfile as sf
        import numpy as np
        data, fs = sf.read(ref_file)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if fs != target_sr:
            num_samples = int(len(data) * target_sr / fs)
            data = np.interp(np.linspace(0, len(data) - 1, num_samples), np.arange(len(data)), data)
        ref_len = len(data)
        if len(wave) < ref_len:
            return False, 0
        
        win_size = int(target_sr * 0.02)
        ref_env = np.convolve(np.abs(data), np.ones(win_size)/win_size, mode='same')
        wave_env = np.convolve(np.abs(wave[:ref_len + int(target_sr * 0.5)]), np.ones(win_size)/win_size, mode='same')
        
        test_len = min(int(target_sr * 3.0), len(ref_env))
        ref_test = ref_env[:test_len]
        max_lag = int(target_sr * 1.0)
        wave_test = wave_env[:test_len + max_lag]
        
        ref_test = (ref_test - np.mean(ref_test)) / (np.std(ref_test) + 1e-8)
        corr = np.correlate(wave_test, ref_test, mode='valid')
        
        best_corr = 0.0
        best_lag = 0
        for lag in range(len(corr)):
            window = wave_env[lag : lag + test_len]
            std_val = np.std(window)
            if std_val > 1e-5:
                norm_window = (window - np.mean(window)) / std_val
                val = np.dot(norm_window, ref_test) / test_len
                if val > best_corr:
                    best_corr = val
                    best_lag = lag
                    
        if best_corr > 0.45:
            return True, ref_len + best_lag
        return False, 0
    except Exception as e:
        return False, 0


# --- Server-Side User Database ---
USERS_FILE = os.path.join(v_tmp, "vox_users.json")

def load_users():
    import json
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    import json
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f)
    except Exception as e:
        print(f"[AUTH] Error saving users: {e}")

class AuthReq(BaseModel):
    name: str = ""
    email: str = ""
    password: str = ""

@app.post("/api/register")
async def register_user(r: AuthReq):
    users = load_users()
    email = r.email.strip().lower()
    if not email or not r.password:
        return {"success": False, "message": "Required fields missing!"}
    if email in users:
        return {"success": False, "message": "Account with this email already exists!"}
    users[email] = {"name": r.name.strip(), "password": r.password}
    save_users(users)
    return {"success": True, "message": "Account created successfully!"}

@app.post("/api/login")
async def login_user(r: AuthReq):
    users = load_users()
    email = r.email.strip().lower()
    if email in users and users[email]["password"] == r.password:
        return {"success": True, "name": users[email]["name"]}
    return {"success": False, "message": "Invalid email or secure key!"}


whisper, f5 = None, None
loading_status = "Initializing Neural Core..."

def load_all():
    global whisper, f5, loading_status
    if not is_kag: 
        loading_status = "LOCAL MODE - NO MODELS"
        return
    try:
        gc.collect(); torch.cuda.empty_cache()
        base = "/kaggle/working"
        loading_status = "Step 1/3: Loading Whisper..."
        from faster_whisper import WhisperModel
        whisper = WhisperModel("large-v3", device="cuda", compute_type="float16", download_root=base)
        ckpt = os.path.join(base, "F5TTS_v1_Base", "model_1250000.safetensors")
        vocab = os.path.join(base, "F5TTS_v1_Base", "vocab.txt")
        if not (os.path.exists(ckpt) and os.path.exists(vocab)):
            snapshot_download(repo_id="SWivid/F5-TTS", allow_patterns=["*.safetensors", "*.json", "*.txt"], local_dir=base, max_workers=1)
        loading_status = "Step 2/3: Fetching F5-TTS Weights..."
        from f5_tts.api import F5TTS
        f5 = F5TTS(device="cuda", ckpt_file=ckpt, vocab_file=vocab)
        gc.collect(); torch.cuda.empty_cache()
        loading_status = "NEURAL CORE 100% READY"
        print("\n[SYSTEM] NEURAL CORE 100% READY!\n")
    except Exception as e: 
        loading_status = f"ERROR: {str(e)[:50]}"
        print(f"[AI] ERROR: {e}")

threading.Thread(target=load_all, daemon=True).start()

class Req(BaseModel):
    speaker_wav_b64: str = ""; text: str = ""; ref_text: str = ""; speed: float = 1.0; volume: float = 1.0; source_audio_b64: str = ""; lang: str = "en"

def dec(s): return base64.b64decode(s.split(',')[1] if ',' in s else s)

def convert_to_standard_wav(input_path, output_path, target_sr=24000):
    try:
        import subprocess
        # ffmpeg command to convert any audio (including webm) to standard 24kHz mono PCM WAV
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1", "-ar", str(target_sr),
            "-c:a", "pcm_s16le", output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            log_debug(f"[ERROR] convert_to_standard_wav ffmpeg failed: {res.stderr}")
            return False
        return True
    except Exception as e:
        log_debug(f"[ERROR] convert_to_standard_wav exception: {e}")
        return False

def get_ref_samples(file_path, target_sr=24000):
    try:
        data, fs = sf.read(file_path)
        duration = len(data) / fs
        return int(duration * target_sr)
    except Exception as e:
        print(f"[ERROR] get_ref_samples failed: {e}")
        return 0

def is_ref_audio_prepended(wave, ref_file, target_sr=24000):
    try:
        data, fs = sf.read(ref_file)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if fs != target_sr:
            num_samples = int(len(data) * target_sr / fs)
            data = np.interp(np.linspace(0, len(data) - 1, num_samples), np.arange(len(data)), data)
        ref_len = len(data)
        if len(wave) < ref_len:
            return False, 0
        
        # Calculate amplitude envelopes to overcome phase shift / vocoder differences
        win_size = int(target_sr * 0.02)
        ref_env = np.convolve(np.abs(data), np.ones(win_size)/win_size, mode='same')
        wave_env = np.convolve(np.abs(wave[:ref_len + int(target_sr * 0.5)]), np.ones(win_size)/win_size, mode='same')
        
        # Test correlation on the first 3 seconds (or full length if shorter)
        test_len = min(int(target_sr * 3.0), len(ref_env))
        ref_test = ref_env[:test_len]
        
        max_lag = int(target_sr * 1.0) # 1 second max delay
        wave_test = wave_env[:test_len + max_lag]
        
        ref_test = (ref_test - np.mean(ref_test)) / (np.std(ref_test) + 1e-8)
        
        corr = np.correlate(wave_test, ref_test, mode='valid')
        best_corr = 0.0
        best_lag = 0
        for lag in range(len(corr)):
            window = wave_env[lag : lag + test_len]
            std_val = np.std(window)
            if std_val > 1e-5:
                norm_window = (window - np.mean(window)) / std_val
                val = np.dot(norm_window, ref_test) / test_len
                if val > best_corr:
                    best_corr = val
                    best_lag = lag
                    
        log_debug(f"Envelope cross-correlation similarity: {best_corr:.4f} at lag {best_lag}")
        if best_corr > 0.35:
            return True, ref_len + best_lag
        return False, ref_len + best_lag
    except Exception as e:
        log_debug(f"is_ref_audio_prepended robust failed: {e}")
        return False, 0

def find_silence_valley(audio, center_idx, search_width, target_sr=24000):
    try:
        start = max(0, center_idx - search_width)
        end = min(len(audio), center_idx + search_width)
        segment = audio[start:end]
        if len(segment) == 0:
            return center_idx
        win_size = int(target_sr * 0.05)
        env = np.convolve(np.abs(segment), np.ones(win_size)/win_size, mode='same')
        min_idx = np.argmin(env)
        return start + min_idx
    except Exception:
        return center_idx

def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    return np.interp(np.linspace(0, len(audio) - 1, num_samples), np.arange(len(audio)), audio)

def get_ref_duration_s(file_path):
    """Returns the duration in seconds of the reference audio file using ffprobe, soundfile, or wave as fallbacks."""
    try:
        # Try ffprobe first as it supports all browser formats (wav, webm, ogg, etc.)
        import subprocess
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            dur = float(res.stdout.strip())
            if dur > 0:
                return dur
    except Exception as e:
        log_debug(f"ffprobe failed in get_ref_duration_s: {e}")
        
    try:
        import soundfile as sf
        data, fs = sf.read(file_path)
        return len(data) / fs
    except Exception as e1:
        log_debug(f"sf.read failed in get_ref_duration_s: {e1}")
        try:
            import wave
            with wave.open(file_path, 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                if rate > 0:
                    return frames / rate
        except Exception as e2:
            log_debug(f"wave.open failed in get_ref_duration_s: {e2}")
    return 0.0

def get_ref_end_timestamp(wave, sr, r_txt):
    """Find the end timestamp of the REFERENCE text in the generated audio.
    
    Why this works better: Target text might be Urdu in Roman English, making
    Whisper transcriptions unpredictable. But r_txt was PREVIOUSLY transcribed
    by Whisper from the source audio, so the spelling is guaranteed to match!
    """
    if not whisper or not r_txt:
        return 0.0
    try:
        wave_16k = resample_audio(wave, sr, 16000)
        segments, _ = whisper.transcribe(wave_16k, word_timestamps=True)
        
        r_words = [re.sub(r'\W+', '', w.lower()) for w in r_txt.split() if re.sub(r'\W+', '', w)]
        if not r_words:
            return 0.0
            
        # Look for the last 3 words of the reference text
        search_len = min(3, len(r_words))
        search_seq = r_words[-search_len:]
        
        all_words = []
        for segment in segments:
            if segment.words:
                all_words.extend(segment.words)
                
        if not all_words: return 0.0
        
        cleaned_all = [re.sub(r'\W+', '', w.word.lower()) for w in all_words]
        log_debug(f"Looking for end of reference sequence: {search_seq}")
        
        # 1. Search for the exact sequence
        for i in range(len(cleaned_all) - search_len + 1):
            match = True
            for j in range(search_len):
                if cleaned_all[i+j] != search_seq[j]:
                    match = False
                    break
            if match:
                end_time = all_words[i + search_len - 1].end
                log_debug(f"Found reference boundary at '{all_words[i + search_len - 1].word}' -> {end_time:.2f}s")
                return end_time
                
        # 2. Fallback: match the very last word of reference, but only within the first 10 seconds
        last_word = search_seq[-1]
        for w_info, cleaned in zip(all_words, cleaned_all):
            if cleaned == last_word and w_info.end < 10.0:
                log_debug(f"Fallback boundary match at '{w_info.word}' -> {w_info.end:.2f}s")
                return w_info.end
                
        log_debug("Could not find reference end in transcription.")
        return 0.0
    except Exception as e:
        log_debug(f"get_ref_end_timestamp failed: {e}")
        return 0.0

def slice_reference_audio(wave, ref_file, target_sr=24000):
    try:
        ref_samples = get_ref_samples(ref_file, target_sr)
        is_prepended, offset = is_ref_audio_prepended(wave, ref_file, target_sr)
        
        # Use offset from correlation only if correlation is strong (is_prepended is True)
        if is_prepended and offset > 0:
            best_offset = offset
            log_debug(f"Using correlation offset: {best_offset}")
        else:
            best_offset = ref_samples
            log_debug(f"Correlation weak/failed. Using fallback ref_samples: {best_offset}")
            
        if best_offset <= 0: return wave
        
        search_width = int(target_sr * 0.8)
        slice_idx = find_silence_valley(wave, best_offset, search_width, target_sr)
        log_debug(f"Slice index found at: {slice_idx} (approx {slice_idx / target_sr:.2f}s)")
        if 0 < slice_idx < len(wave): return wave[slice_idx:]
        return wave
    except Exception as e:
        log_debug(f"slice_reference_audio failed: {e}")
        return wave

def trim_leaked_reference_speech(wave, target_sr=24000):
    try:
        max_search_len = int(target_sr * 2.5)
        if len(wave) < max_search_len:
            max_search_len = len(wave)
        win_size = int(target_sr * 0.05)
        if win_size == 0:
            return wave
        env = np.convolve(np.abs(wave[:max_search_len]), np.ones(win_size)/win_size, mode='same')
        env_min = np.min(env)
        env_max = np.max(env) + 1e-8
        norm_env = (env - env_min) / (env_max - env_min)
        edge = int(target_sr * 0.1)
        if len(norm_env) <= 2 * edge:
            return wave
        search_region = norm_env[edge:-edge]
        min_idx = np.argmin(search_region) + edge
        min_val = norm_env[min_idx]
        energy_before = np.max(norm_env[:min_idx])
        energy_after = np.max(norm_env[min_idx:])
        log_debug(f"Leakage detection: min_val={min_val:.4f}, before={energy_before:.4f}, after={energy_after:.4f} at sample {min_idx}")
        
        # Relative energy check: the valley must be significantly quieter than both the preceding and succeeding segments.
        is_leakage = (min_val < 0.35 and 
                      min_val < 0.5 * min(energy_before, energy_after) and 
                      energy_before > 0.25 and 
                      energy_after > 0.25)
                      
        if is_leakage:
            log_debug(f"Trimming leaked reference speech at sample {min_idx} (approx {min_idx / target_sr:.2f}s)")
            return wave[min_idx:]
        return wave
    except Exception as e:
        log_debug(f"trim_leaked_reference_speech failed: {e}")
        return wave


def robust_slice_reference_audio(wave, sr, ref_file, ref_text, target_text, speed):
    try:
        # 1. Get exact duration using our ffprobe fallback (fixes WebM 0.0s bug)
        ref_dur_s = get_ref_duration_s(ref_file)
        if ref_dur_s <= 0.0:
            log_debug("[SLICE] Reference duration is 0, skipping slice.")
            return wave
            
        # 2. Exact Slice:
        # We will lock the neural model speed to 1.0 to prevent hallucinations.
        # This means the reference audio is exactly its original length.
        expected_ref_samples = int(ref_dur_s * sr)
        
        # 3. Add a tiny search for the nearest "silence valley" 
        # This acts like a smart cross-fade to prevent "clicking" sounds at the cut point.
        search_width = int(sr * 0.15) # 0.15s search window
        slice_idx = find_silence_valley(wave, expected_ref_samples, search_width, sr)
        
        # Safety bounds
        min_safe_idx = max(0, expected_ref_samples - int(sr * 0.3))
        max_safe_idx = max(0, len(wave) - int(sr * 0.5))
        
        if slice_idx < min_safe_idx:
            slice_idx = min(expected_ref_samples, max_safe_idx)
            
        if 0 < slice_idx < len(wave):
            log_debug(f"[SLICE] Official Technique applied. Sliced at {slice_idx} samples ({slice_idx / sr:.2f}s).")
            wave = wave[slice_idx:]
            
        return wave
    except Exception as e:
        log_debug(f"[SLICE] robust_slice_reference_audio failed: {e}")
        return wave


@app.get("/status")
async def get_status():
    return {"status": loading_status, "ready": (f5 is not None)}

@app.post("/autofix")
async def afix(r: Req):
    t = r.text
    # English Shorthands
    en_slangs = {r"\bnut\b":"not", r"\bgud\b":"good", r"\bv\b":"we", r"\bu\b":"you", r"\br\b":"are", r"\by\b":"why", r"\bdat\b":"that", r"\bdis\b":"this", r"\bplz\b":"please", r"\bsry\b":"sorry", r"\bthx\b":"thanks", r"\bbcz\b":"because", r"\bbro\b":"brother", r"\bur\b":"your", r"\bidk\b":"I don't know", r"\bomg\b":"oh my god"}
    for k, v in en_slangs.items(): t = re.sub(k, v, t, flags=re.IGNORECASE)
    
    # Urdu/Hindi Shorthands
    if r.lang != "en":
        ur_slangs = {r"\bkia\b":"kiya", r"\bh\b":"hai", r"\bb\b":"bhi", r"\bn\b":"nahi", r"\bmne\b":"maine", r"\bmjhe\b":"mujhe", r"\bkse\b":"kaise", r"\byh\b":"yeh", r"\bhy\b":"hai", r"\brha\b":"raha", r"\brhi\b":"rahi", r"\bsy\b":"se", r"\bha\b":"haan", r"\bthk\b":"theek", r"\bdia\b":"diya", r"\bkro\b":"karo", r"\bkr\b":"kar"}
        for k, v in ur_slangs.items(): t = re.sub(k, v, t, flags=re.IGNORECASE)
    
    # Auto-capitalize first letters
    final_t = "".join([s.capitalize() for s in re.split('([.!?] *)', t)]).strip()
    if final_t and not final_t.endswith(('.', '!', '?')): final_t += "."
    return {"text": final_t}

@app.post("/transcribe")
async def tr(r: Req):
    if not whisper: return {"text": "Loading..."}
    p_raw = os.path.join(v_tmp, "tr_raw")
    p = os.path.join(v_tmp, "tr.wav")
    with open(p_raw, "wb") as f: f.write(dec(r.speaker_wav_b64))
    if not convert_to_standard_wav(p_raw, p):
        shutil.copy(p_raw, p)
    s, _ = whisper.transcribe(p, language=r.lang)
    return {"text": " ".join([i.text for i in s]).strip()}

# Duplicated endpoints removed. Updated versions with semantic slicing are active at the bottom.

UI_HTML = """<!DOCTYPE html><html><head><title>VOX CLONE MASTERPIECE</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;600;900&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
:root {
    --bg: #03030a;
    --surface: rgba(12, 10, 22, 0.55);
    --border: rgba(139, 92, 246, 0.12);
    --border-hover: rgba(139, 92, 246, 0.45);
    --accent: #8b5cf6;
    --accent2: #d946ef;
    --glow: rgba(139, 92, 246, 0.18);
    --text: #c4c4d0;
    --text-dim: rgba(200, 200, 220, 0.28);
}
body { background: var(--bg); color: var(--text); font-family: 'Outfit', sans-serif; margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center; overflow-x: hidden; }
#neuralCanvas { position: fixed; inset: 0; z-index: -1; opacity: 1.0; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 10px; }

.app { position: relative; z-index: 10; width: 100%; max-width: 860px; padding: 28px 22px 48px; box-sizing: border-box; }
h1 { font-size: 3.5rem; font-weight: 900; text-align: center; letter-spacing: -2px; color: #fff; text-shadow: 0 0 15px var(--glow); margin: 0 0 25px 0; animation: float 4s ease-in-out infinite; }
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-12px); } }

#st-bar { text-align: center; font-family: 'JetBrains Mono'; font-size: 0.68rem; color: var(--accent); margin-bottom: 22px; background: rgba(139,92,246,0.06); padding: 11px 20px; border-radius: 14px; border: 1px solid var(--border); transition: 0.4s; }
#st-bar.ready { background: rgba(139,92,246,0.18); color: #fff; box-shadow: 0 0 18px var(--glow); border-color: rgba(139,92,246,0.5); }

.card { background: var(--surface); border: 1px solid var(--border); border-radius: 26px; padding: 24px; margin-bottom: 20px; backdrop-filter: blur(80px); transition: 0.4s; }
.card:hover { border-color: var(--border-hover); box-shadow: 0 8px 35px var(--glow); transform: translateY(-2px); }
.label { font-family: 'JetBrains Mono'; font-size: 0.58rem; color: var(--text-dim); letter-spacing: 4px; text-transform: uppercase; display: block; margin-bottom: 11px; }

input[type="file"] { background: rgba(139, 92, 246, 0.08); color: #d8b4fe; border: 1px solid rgba(139, 92, 246, 0.2); padding: 11px 14px; border-radius: 14px; width: 100%; height: 46px; box-sizing: border-box; font-size: 0.82rem; transition: 0.3s; cursor: pointer; }
input[type="file"]:hover { border-color: rgba(139, 92, 246, 0.5); box-shadow: 0 0 12px var(--glow); }
input[type="file"]::file-selector-button { display: none; }
.btn-action { background: transparent; border: 1px solid rgba(139,92,246,0.15); color: var(--text); width: 46px; height: 46px; border-radius: 13px; cursor: pointer; transition: 0.3s; font-size: 1.1rem; display: flex; align-items: center; justify-content: center; }
.btn-action:hover { background: var(--accent); color: #fff; box-shadow: 0 5px 14px var(--glow); }
.btn-mic.recording { background: #ef4444; border-color: #ef4444; color: #fff; }

.text-box { position: relative; width: 100%; margin-bottom: 18px; }
textarea { width: 100%; background: rgba(0,0,0,0.32); border: 1px solid rgba(255,255,255,0.04); color: var(--text); padding: 15px 16px; border-radius: 18px; font-family: 'JetBrains Mono'; font-size: 0.88rem; outline: none; box-sizing: border-box; resize: none; transition: 0.3s; }
textarea:hover { border-color: rgba(139, 92, 246, 0.3); }
textarea:focus, input:focus, select:focus { border-color: var(--accent); box-shadow: 0 0 18px var(--glow); background: rgba(8, 6, 20, 0.65); }
.btn-magic { position: absolute; bottom: 9px; right: 9px; width: 34px; height: 34px; font-size: 0.95rem; border-radius: 10px; background: transparent; border: 1px solid rgba(139, 92, 246, 0.25); color: var(--text-dim); cursor: pointer; transition: 0.3s; }
.btn-magic:hover { background: var(--accent); color: #fff; transform: scale(1.1) rotate(8deg); }

.speed-box { flex: 0.8; min-width: 190px; background: rgba(139, 92, 246, 0.04); padding: 14px 16px; border-radius: 16px; border: 1px solid var(--border); transition: 0.3s; }
.speed-box:hover { border-color: rgba(139, 92, 246, 0.4); box-shadow: 0 0 15px rgba(139, 92, 246, 0.12); }
.speed-slider { width: 100%; accent-color: var(--accent); height: 5px; margin-top: 8px; }
select { width: 100%; background: rgba(0,0,0,0.32); border: 1px solid rgba(255,255,255,0.04); color: var(--text); padding: 12px 14px; border-radius: 14px; font-family: 'JetBrains Mono'; font-size: 0.85rem; outline: none; transition: 0.3s; }
select:hover { border-color: rgba(139, 92, 246, 0.4); }
option { background: #12101e; color: var(--text); font-family: 'Outfit', sans-serif; }

.btn-ultra { width: 100%; padding: 20px; background: linear-gradient(135deg, var(--accent), var(--accent2)); border: none; border-radius: 20px; color: #fff; font-weight: 900; font-size: 1.15rem; cursor: pointer; transition: 0.4s; letter-spacing: 4px; text-transform: uppercase; }
.btn-ultra:hover { transform: translateY(-3px); box-shadow: 0 14px 35px rgba(139, 92, 246, 0.5); }
.tabs { display: flex; gap: 8px; margin-bottom: 24px; background: rgba(255,255,255,0.015); padding: 6px; border-radius: 20px; border: 1px solid rgba(139,92,246,0.06); }
.tab { flex: 1; padding: 13px 8px; border: none; border-radius: 16px; background: transparent; color: #a78bfa; font-weight: 700; cursor: pointer; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 2px; transition: 0.3s; }
.tab:hover { background: rgba(139,92,246,0.1); }
.tab.active { background: linear-gradient(135deg, var(--accent), #6366f1); color: #fff; box-shadow: 0 4px 18px var(--glow); }

audio { width: 100%; margin-top: 8px; height: 50px; border-radius: 50px; outline: none; transition: 0.3s; }
audio::-webkit-media-controls-panel { background-color: #e8e2f8; transition: 0.3s; }
audio::-webkit-media-controls-current-time-display, audio::-webkit-media-controls-time-remaining-display { color: #1e1b4b; font-weight: 700; text-shadow: none; font-size: 0.95rem; }
.output-area { margin-top: 18px; display: flex; flex-direction: column; width: 100%; }
.processing-bay { flex-direction: column; align-items: center; margin-bottom: 12px; gap: 10px; width: 100%; }
.mini-ring { width: 28px; height: 28px; border: 2.5px solid rgba(139,92,246,0.1); border-top: 2.5px solid var(--accent); border-radius: 50%; animation: spin 1s linear infinite; }
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

/* --- PREMIUM AUTH PORTAL STYLES --- */
#auth-container {
    max-width: 440px;
    width: 90%;
    margin: 40px auto;
    padding: 35px;
    border-radius: 28px;
    background: rgba(10, 10, 15, 0.45);
    border: 1px solid rgba(139, 92, 246, 0.15);
    backdrop-filter: blur(40px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.5), 0 0 20px rgba(139, 92, 246, 0.05);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
#auth-container:hover {
    border-color: rgba(139, 92, 246, 0.35);
    box-shadow: 0 20px 50px rgba(139, 92, 246, 0.15);
}

.auth-header {
    text-align: center;
    margin-bottom: 30px;
}
.auth-title {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -1px;
    color: #fff;
    margin: 0;
    text-shadow: 0 0 15px var(--glow);
}
.auth-title span {
    background: linear-gradient(135deg, #8b5cf6, #d946ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 20px rgba(139, 92, 246, 0.5);
}
.auth-subtitle {
    font-size: 0.80rem;
    color: rgba(255, 255, 255, 0.4);
    margin-top: 5px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 2px;
}

.input-group {
    position: relative;
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
}

.input-group label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: rgba(255, 255, 255, 0.35);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 8px;
    transition: color 0.3s;
}

.input-group:focus-within label {
    color: #8b5cf6;
}

.input-field {
    width: 100%;
    background: rgba(0, 0, 0, 0.45);
    border: 1px solid rgba(139, 92, 246, 0.1);
    color: #fff;
    padding: 14px 18px;
    border-radius: 16px;
    font-family: 'Outfit', sans-serif;
    font-size: 0.95rem;
    outline: none;
    box-sizing: border-box;
    transition: all 0.3s ease;
}
.input-field:hover {
    border-color: rgba(139, 92, 246, 0.35);
    background: rgba(10, 10, 15, 0.6);
}
.input-field:focus {
    border-color: #8b5cf6;
    box-shadow: 0 0 15px var(--glow);
    background: rgba(10, 10, 20, 0.7);
}

.password-wrapper {
    position: relative;
    width: 100%;
}
.password-toggle {
    position: absolute;
    right: 15px;
    top: 50%;
    transform: translateY(-50%);
    background: none;
    border: none;
    color: rgba(255, 255, 255, 0.4);
    cursor: pointer;
    font-size: 1.1rem;
    padding: 0;
    transition: color 0.3s;
}
.password-toggle:hover {
    color: #a78bfa;
}

/* Height & Slide Transition for Name Field */
#signup-fields-wrapper {
    max-height: 0;
    opacity: 0;
    overflow: hidden;
    transition: max-height 0.35s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease;
}
#signup-fields-wrapper.active {
    max-height: 100px;
    opacity: 1;
}

/* Animated Alert/Message inside Portal */
.alert-banner {
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 20px;
    display: none;
    align-items: center;
    gap: 10px;
    animation: slideDown 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}
.alert-error {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.25);
    color: #fc8181;
}
.alert-success {
    background: rgba(72, 187, 120, 0.12);
    border: 1px solid rgba(72, 187, 120, 0.25);
    color: #68d391;
}

.auth-btn {
    width: 100%;
    padding: 16px;
    background: linear-gradient(135deg, #8b5cf6, #d946ef);
    border: none;
    border-radius: 16px;
    color: #fff;
    font-weight: 800;
    font-size: 1.05rem;
    cursor: pointer;
    transition: all 0.3s ease;
    letter-spacing: 2px;
    text-transform: uppercase;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2);
}
.auth-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
}
.auth-btn:active {
    transform: translateY(0);
}

.auth-footer {
    text-align: center;
    margin-top: 25px;
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.4);
}
.auth-footer a {
    color: #a78bfa;
    text-decoration: none;
    font-weight: 600;
    margin-left: 5px;
    transition: color 0.3s;
}
.auth-footer a:hover {
    color: #d946ef;
    text-shadow: 0 0 10px rgba(217, 70, 239, 0.3);
}

/* Logout Button */
.logout-btn {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.6);
    padding: 8px 16px;
    border-radius: 12px;
    cursor: pointer;
    font-size: 0.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    transition: all 0.3s;
}
.logout-btn:hover {
    background: rgba(239, 68, 68, 0.2);
    border-color: rgba(239, 68, 68, 0.4);
    color: #fff;
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);
}

/* --- MOBILE RESPONSIVENESS STYLES --- */
@media (max-width: 600px) {
    h1 {
        font-size: 2.2rem !important;
        margin-bottom: 15px !important;
    }
    .app {
        padding: 12px !important;
    }
    .card {
        padding: 16px !important;
        border-radius: 20px !important;
        margin-bottom: 15px !important;
    }
    #auth-container {
        padding: 24px 18px !important;
        margin: 20px auto !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    .auth-title {
        font-size: 2.0rem !important;
    }
    .dashboard-header {
        flex-direction: column !important;
        gap: 12px !important;
        margin-bottom: 20px !important;
    }
    .dashboard-header .header-title {
        margin-left: 0 !important;
        text-align: center !important;
    }
    .logout-btn {
        width: 100% !important;
        text-align: center !important;
        padding: 10px !important;
    }
    .speed-box {
        width: 100% !important;
        box-sizing: border-box !important;
    }
}
</style></head>
<body><canvas id="neuralCanvas"></canvas>
<div class="app">
    <!-- Auth Portal (Login / Signup) -->
    <div id="auth-container">
        <div class="auth-header">
            <h2 class="auth-title">VOX <span>CLONE</span></h2>
            <div id="auth-subtitle" class="auth-subtitle">INITIALIZE ACCOUNT</div>
        </div>

        <div id="alert-banner" class="alert-banner"></div>

        <div id="signup-fields-wrapper" class="active">
            <div class="input-group">
                <label for="auth-name">Full Name</label>
                <input type="text" id="auth-name" class="input-field" placeholder="">
            </div>
        </div>

        <div class="input-group">
            <label for="auth-email">Email Address</label>
            <input type="email" id="auth-email" class="input-field" placeholder="">
        </div>

        <div class="input-group">
            <label for="auth-pass">Secure Key</label>
            <div class="password-wrapper">
                <input type="password" id="auth-pass" class="input-field" placeholder="">
                <button type="button" class="password-toggle" onclick="togglePasswordVisibility()">👁️</button>
            </div>
        </div>

        <button id="auth-submit-btn" class="auth-btn" onclick="handleAuth()">Initialize Account</button>

        <div class="auth-footer">
            <span id="auth-toggle-text">Already exists?</span>
            <a href="#" id="auth-toggle-link" onclick="toggleAuthMode(event)">Access Core</a>
        </div>
    </div>

    <!-- Main Vox Clone Content (Initially Hidden with opacity and slide transitions) -->
    <div id="main-app-content" style="display: none; transition: opacity 0.5s ease-out, transform 0.5s ease-out; transform: translateY(20px); opacity: 0;">
        <h1 class="dashboard-header" style="display: flex; align-items: center; justify-content: space-between; gap: 15px; margin-bottom: 25px; position: relative;">
            <span class="header-title" style="flex: 1; text-align: center; font-size: clamp(2.4rem, 6vw, 60px); font-weight: 900; letter-spacing: -3px; color: #fff; margin: 0; text-shadow: 0 0 40px var(--glow); animation: float 5s ease-in-out infinite;">VOX <span style="background: linear-gradient(135deg, #8b5cf6, #d946ef); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">CLONE</span></span>
            <button class="logout-btn" onclick="logout()">LOGOUT</button>
        </h1>
        <div id="st-bar">NEURAL CORE SYNCHRONIZING...</div>
        <div class="tabs"><button class="tab active" onclick="st('clone', this)">CLONING</button><button class="tab" onclick="st('morph', this)">MORPHER</button></div>
        <div id="p-clone" class="panel"><div class="card"><span class="label">Neural DNA</span><div style="display:flex; align-items:center; gap:8px; margin-bottom:18px;"><button class="btn-action btn-mic" id="m1" onclick="rec('f1','m1')">🎙️</button><input type="file" id="f1" onchange="tr()"></div><div class="text-box"><textarea id="t1" placeholder="Voice analysis..." rows="2"></textarea></div></div><div class="card" id="c2"><span class="label">Synthesis Matrix</span><div class="text-box"><textarea id="t2" placeholder="Input Script..." rows="3"></textarea><button class="btn-magic" onclick="af('t2')">🪄</button></div><div style="display:flex; justify-content:space-between; align-items:flex-end; gap:12px; margin-bottom:18px; flex-wrap:wrap;"><div style="flex:1; min-width:130px;"><span class="label">Domain</span><select id="l1"><option value="en">English</option><option value="ur">Urdu</option><option value="hi">Hindi</option><option value="zh">Chinese</option><option value="ja">Japanese</option><option value="ko">Korean</option><option value="es">Spanish</option><option value="fr">French</option><option value="de">German</option><option value="pt">Portuguese</option><option value="ru">Russian</option></select></div><div class="speed-box"><span class="label">Speed: <span id="sv1" style="color:#8b5cf6;">1.0</span>x</span><input type="range" id="s1" class="speed-slider" min="0.0" max="2.0" step="0.1" value="1.0" oninput="document.getElementById('sv1').innerText=this.value"></div></div><div style="display:flex; flex-direction:column;"><button class="btn-ultra" onclick="runCl()">ACTIVATE SYNTHESIS</button><div class="output-area"><div class="processing-bay" style="display:none;"><div class="mini-ring"></div><div class="mini-text" style="font-family:'JetBrains Mono'; font-size:0.5rem; letter-spacing:2px;">Neural Processing...</div></div><audio id="a1" controls controlsList="noplaybackrate"></audio></div></div></div></div>
        <div id="p-morph" class="panel" style="display:none;"><div class="card"><span class="label">Target DNA</span><div style="display:flex; align-items:center; gap:8px; margin-bottom:18px;"><button class="btn-action btn-mic" id="m2" onclick="rec('fd','m2')">🎙️</button><input type="file" id="fd"></div></div><div class="card" id="c4"><span class="label">Source Speech</span><div style="display:flex; align-items:center; gap:8px; margin-bottom:18px;"><button class="btn-action btn-mic" id="m3" onclick="rec('fs','m3')">🎙️</button><input type="file" id="fs"></div><div style="gap:10px; flex-wrap:wrap; margin-bottom:18px;"><div class="speed-box"><span class="label">Speed: <span id="sv2" style="color:#8b5cf6;">1.0</span>x</span><input type="range" id="s2" class="speed-slider" min="0.0" max="2.0" step="0.1" value="1.0" oninput="document.getElementById('sv2').innerText=this.value"></div></div><button class="btn-ultra" style="margin-top: 12px;" onclick="runMo()">IDENTITY SWAP</button><div class="output-area"><div class="processing-bay" style="display:none;"><div class="mini-ring"></div><div class="mini-text" style="font-family:'JetBrains Mono'; font-size:0.5rem; letter-spacing:2px;">Swapping Identities...</div></div><audio id="a2" controls controlsList="noplaybackrate"></audio></div></div></div></div>
        
    </div>
</div><script>
    const canvas=document.getElementById('neuralCanvas'), ctx=canvas.getContext('2d');
    let pts=[]; function init_bg(){ canvas.width=window.innerWidth; canvas.height=window.innerHeight; pts=[]; for(let i=0;i<40;i++) pts.push({x:Math.random()*canvas.width, y:Math.random()*canvas.height, vx:(Math.random()-0.5)*0.8, vy:(Math.random()-0.5)*0.8, s:Math.random()*2+0.8}); }
    init_bg(); window.onresize=init_bg;
    function anim(){ ctx.fillStyle='rgba(3, 3, 5, 0.28)'; ctx.fillRect(0,0,canvas.width,canvas.height); pts.forEach((p,i)=>{ p.x+=p.vx; p.y+=p.vy; if(p.x<0||p.x>canvas.width)p.vx*=-1; if(p.y<0||p.y>canvas.height)p.vy*=-1; ctx.beginPath(); ctx.arc(p.x,p.y,p.s,0,Math.PI*2); ctx.fillStyle=`hsla(260, 75%, 65%, 0.65)`; ctx.fill(); for(let j=i+1;j<pts.length;j++){ let d=Math.hypot(p.x-pts[j].x, p.y-pts[j].y); if(d<150){ ctx.strokeStyle=`rgba(139,92,246,${0.4*(1-d/150)})`; ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(pts[j].x,pts[j].y); ctx.stroke(); } } }); requestAnimationFrame(anim); } anim();
    function st(t, el){document.querySelectorAll('.panel').forEach(p=>p.style.display='none'); document.querySelectorAll('.tab').forEach(tb=>tb.classList.remove('active')); document.getElementById('p-'+t).style.display='block'; el.classList.add('active');}
    
    setInterval(async () => {
        try {
            const r = await fetch('/status');
            const data = await r.json();
            const bar = document.getElementById('st-bar');
            bar.innerText = data.status;
            if(data.ready) bar.classList.add('ready');
            else bar.classList.remove('ready');
        } catch(e) {}
    }, 2500);

    async function b6(f){ return new Promise((res, rej) => { const rd = new FileReader(); rd.readAsDataURL(f); rd.onload = () => res(rd.result.split(',')[1]); rd.onerror = e => rej(e); }); }
    async function af(id){ const el = document.getElementById(id); let t = el.value.trim(); if(!t) return; try { const r = await fetch('/autofix', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text:t, lang:document.getElementById('l1').value})}); const data = await r.json(); el.value = data.text; } catch(e) { console.error(e); } }
    let R=null, ch=[];
    async function rec(fid, mid){ const b=document.getElementById(mid); if(R && R.state==='recording'){ R.stop(); b.classList.remove('recording'); b.innerText='🎙️'; return; } const s=await navigator.mediaDevices.getUserMedia({audio:true}); R=new MediaRecorder(s); ch=[]; R.ondataavailable=e=>ch.push(e.data); R.onstop=async()=>{ const bl=new Blob(ch,{type:'audio/wav'}); const f=new File([bl], "rec.wav", {type:'audio/wav'}); const dt=new DataTransfer(); dt.items.add(f); const inp=document.getElementById(fid); inp.files=dt.files; inp.dispatchEvent(new Event('change')); }; R.start(); b.classList.add('recording'); b.innerText='⏹️'; }
    async function tr(){ const f=document.getElementById('f1').files[0]; if(!f) return; const b6data=await b6(f); const r=await fetch('/transcribe', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({speaker_wav_b64:b6data, lang:document.getElementById('l1').value})}); const data=await r.json(); document.getElementById('t1').value=data.text; }
    async function runCl(){ const c=document.getElementById('c2'); const f=document.getElementById('f1').files[0], t=document.getElementById('t2').value; if(!f || !t){ alert("Data Missing!"); return; } const bay = c.closest('.panel').querySelector('.processing-bay'); bay.style.display='flex'; try { const b6data=await b6(f); const vol=1.0; const uiSpeed=parseFloat(document.getElementById('s1').value); const r=await fetch('/clone',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t,ref_text:document.getElementById('t1').value,speed:uiSpeed,volume:vol,lang:document.getElementById('l1').value,speaker_wav_b64:b6data})}); if(!r.ok){ alert("Engine Error: " + await r.text()); return; } const bl=await r.blob(); const a=document.getElementById('a1'); a.src=URL.createObjectURL(bl); a.play(); } catch(e){alert(e);} finally{bay.style.display='none';} }
    async function runMo(){ const c=document.getElementById('c4'); const tf=document.getElementById('fd').files[0], sf=document.getElementById('fs').files[0]; if(!tf || !sf){ alert("Data Missing!"); return; } const bay = c.querySelector('.processing-bay'); bay.style.display='flex'; try { const tB6=await b6(tf), sB6=await b6(sf); const vol=1.0; const uiSpeed=parseFloat(document.getElementById('s2').value); const r=await fetch('/morph',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({speaker_wav_b64:tB6, source_audio_b64:sB6, speed:uiSpeed, volume:vol, lang:document.getElementById('l1').value})}); if(!r.ok){ alert("Engine Error: " + await r.text()); return; } const bl=await r.blob(); const a=document.getElementById('a2'); a.src=URL.createObjectURL(bl); a.play(); } catch(e){alert(e);} finally{bay.style.display='none';} }

    // --- USER SESSION & AUTHENTICATION PORTAL LOGIC ---
    let authMode = "signup";
    
    function showAlert(msg, type) {
        const banner = document.getElementById('alert-banner');
        banner.innerText = msg;
        banner.className = 'alert-banner ' + (type === 'success' ? 'alert-success' : 'alert-error');
        banner.style.display = 'flex';
    }
    
    function hideAlert() {
        document.getElementById('alert-banner').style.display = 'none';
    }

    function togglePasswordVisibility() {
        const p = document.getElementById('auth-pass');
        const btn = document.querySelector('.password-toggle');
        if (p.type === 'password') {
            p.type = 'text';
            btn.innerText = '🔒';
        } else {
            p.type = 'password';
            btn.innerText = '👁️';
        }
    }

    function checkSession() {
        const currentUser = localStorage.getItem('current_user');
        const authContainer = document.getElementById('auth-container');
        const mainAppContent = document.getElementById('main-app-content');
        
        if (currentUser) {
            // Fade out auth panel
            authContainer.style.opacity = '0';
            authContainer.style.transform = 'scale(0.95)';
            setTimeout(() => {
                authContainer.style.display = 'none';
                mainAppContent.style.display = 'block';
                setTimeout(() => {
                    mainAppContent.style.opacity = '1';
                    mainAppContent.style.transform = 'translateY(0)';
                }, 50);
            }, 300);
        } else {
            // Fade out main content
            mainAppContent.style.opacity = '0';
            mainAppContent.style.transform = 'translateY(20px)';
            setTimeout(() => {
                mainAppContent.style.display = 'none';
                authContainer.style.display = 'block';
                setTimeout(() => {
                    authContainer.style.opacity = '1';
                    authContainer.style.transform = 'scale(1)';
                }, 50);
            }, 300);
        }
    }

    // Run session check on page load
    checkSession();

    function toggleAuthMode(e) {
        if (e) e.preventDefault();
        hideAlert();
        const signupWrapper = document.getElementById('signup-fields-wrapper');
        const authSubtitle = document.getElementById('auth-subtitle');
        const submitBtn = document.getElementById('auth-submit-btn');
        const toggleText = document.getElementById('auth-toggle-text');
        const toggleLink = document.getElementById('auth-toggle-link');
        
        if (authMode === "login") {
            authMode = "signup";
            signupWrapper.classList.add('active');
            authSubtitle.innerText = "INITIALIZE ACCOUNT";
            submitBtn.innerText = "Initialize Account";
            toggleText.innerText = "Already exists?";
            toggleLink.innerText = "Access Core";
        } else {
            authMode = "login";
            signupWrapper.classList.remove('active');
            authSubtitle.innerText = "ACCESS PORTAL";
            submitBtn.innerText = "Access Core";
            toggleText.innerText = "New to Vox?";
            toggleLink.innerText = "Initialize Account";
        }
    }

    async function handleAuth() {
        hideAlert();
        const email = document.getElementById('auth-email').value.trim();
        const pass = document.getElementById('auth-pass').value;
        
        if (!email || !pass) {
            showAlert("Please fill in all required fields!", "error");
            return;
        }
        
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
            showAlert("Please enter a valid email address!", "error");
            return;
        }
        
        if (authMode === "signup") {
            const name = document.getElementById('auth-name').value.trim();
            if (!name) {
                showAlert("Please enter your name!", "error");
                return;
            }
            try {
                const r = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: name, email: email, password: pass })
                });
                const data = await r.json();
                if (data.success) {
                    localStorage.setItem('current_user', email);
                    showAlert("Account created successfully! Connecting...", "success");
                    setTimeout(() => {
                        checkSession();
                        document.getElementById('auth-name').value = "";
                        document.getElementById('auth-email').value = "";
                        document.getElementById('auth-pass').value = "";
                    }, 1000);
                } else {
                    showAlert(data.message || "Registration failed!", "error");
                }
            } catch(e) {
                showAlert("Server connection error!", "error");
            }
        } else {
            try {
                const r = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: email, password: pass })
                });
                const data = await r.json();
                if (data.success) {
                    localStorage.setItem('current_user', email);
                    showAlert("Access granted! Syncing neural link...", "success");
                    setTimeout(() => {
                        checkSession();
                        document.getElementById('auth-email').value = "";
                        document.getElementById('auth-pass').value = "";
                    }, 1000);
                } else {
                    showAlert(data.message || "Invalid email or secure key!", "error");
                }
            } catch(e) {
                showAlert("Server connection error!", "error");
            }
        }
    }

    function logout() {
        localStorage.removeItem('current_user');
        checkSession();
    }
</script></body></html>"""

# --- 4. CLOUDFLARE AUTO-TUNNEL ---
def start_tunnel():
    if not is_kag: return
    c_cmd = os.path.join(v_tmp, "cloudflared")
    if not os.path.exists(c_cmd):
        subprocess.run(["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", c_cmd])
        subprocess.run(["chmod", "+x", c_cmd])
    proc = subprocess.Popen([c_cmd, "tunnel", "--url", "http://127.0.0.1:8000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if "trycloudflare.com" in line:
            url = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if url: print(f"\n💎 CLOUDFLARE URL: {url.group(0)}\n"); break

def start_localtunnel():
    if not is_kag: return
    time.sleep(10) # Wait for server to boot
    proc = subprocess.Popen(["lt", "--port", "8000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if "your url is" in line.lower():
            print(f"\n🚀 LOCALTUNNEL URL: {line.split('is:')[-1].strip()}\n"); break

threading.Thread(target=start_tunnel, daemon=True).start()
threading.Thread(target=start_localtunnel, daemon=True).start()

@app.post("/clone")
async def cl(r: Req):
    if not f5: return Response(content="Loading...", status_code=503)
    try:
        gc.collect(); torch.cuda.empty_cache()
        t_id = int(time.time() * 1000)
        r_p_raw = os.path.join(v_tmp, f"r_raw_{t_id}")
        r_p = os.path.join(v_tmp, f"r_{t_id}.wav")
        with open(r_p_raw, "wb") as f: f.write(dec(r.speaker_wav_b64))
        
        if not convert_to_standard_wav(r_p_raw, r_p):
            shutil.copy(r_p_raw, r_p)
        
        r_txt = r.ref_text.strip()
        if not r_txt or len(r_txt) < 2:
            if whisper:
                s, _ = whisper.transcribe(r_p, language=r.lang)
                r_txt = " ".join([i.text for i in s]).strip()
        actual_speed = r.speed
        wave, sr, _ = f5.infer(ref_file=r_p, ref_text=r_txt or "Hello.", gen_text=r.text.strip(), speed=actual_speed, nfe_step=64, cfg_strength=2.0)
        
        is_prepended, offset = is_ref_audio_prepended(wave, r_p, sr)
        if is_prepended:
            if 0 < offset < len(wave):
                wave = wave[offset:]
        
        p_out = os.path.join(v_tmp, f"o_{t_id}.wav")
        sf.write(p_out, wave, sr)
        p_final = os.path.join(v_tmp, f"clone_{t_id}.wav")
        subprocess.run(["ffmpeg", "-i", p_out, "-filter:a", f"volume={r.volume}", "-y", p_final], capture_output=True)
        gc.collect(); torch.cuda.empty_cache()
        return FileResponse(p_final, media_type="audio/wav")
    except Exception as e: return Response(content=str(e), status_code=500)

@app.post("/morph")
async def morph(r: Req):
    if not f5 or not whisper: return Response(content="Loading...", status_code=503)
    try:
        gc.collect(); torch.cuda.empty_cache()
        t_id = int(time.time() * 1000)
        tw_raw = os.path.join(v_tmp, f"tw_raw_{t_id}")
        sw_raw = os.path.join(v_tmp, f"sw_raw_{t_id}")
        tw = os.path.join(v_tmp, f"tw_{t_id}.wav")
        sw = os.path.join(v_tmp, f"sw_{t_id}.wav")
        with open(tw_raw, "wb") as f: f.write(dec(r.speaker_wav_b64))
        with open(sw_raw, "wb") as f: f.write(dec(r.source_audio_b64))
        
        if not convert_to_standard_wav(tw_raw, tw):
            shutil.copy(tw_raw, tw)
        if not convert_to_standard_wav(sw_raw, sw):
            shutil.copy(sw_raw, sw)
            
        s_res, _ = whisper.transcribe(sw, task="translate"); sw_txt = " ".join([i.text for i in s_res]).strip()
        t_res, _ = whisper.transcribe(tw, task="translate"); tw_txt = " ".join([i.text for i in t_res]).strip()
        
        actual_speed = r.speed
        wave, sr, _ = f5.infer(ref_file=tw, ref_text=tw_txt or "Hello.", gen_text=sw_txt, speed=actual_speed, nfe_step=64, cfg_strength=2.0)
        
        is_prepended, offset = is_ref_audio_prepended(wave, tw, sr)
        if is_prepended:
            if 0 < offset < len(wave):
                wave = wave[offset:]
        
        p_out = os.path.join(v_tmp, f"mo_{t_id}.wav")
        sf.write(p_out, wave, sr)
        p_final = os.path.join(v_tmp, f"morph_{t_id}.wav")
        subprocess.run(["ffmpeg", "-i", p_out, "-filter:a", f"volume={r.volume}", "-y", p_final], capture_output=True)
        gc.collect(); torch.cuda.empty_cache()
        return FileResponse(p_final, media_type="audio/wav")
    except Exception as e: return Response(content=str(e), status_code=500)

@app.get("/")
async def root(): return HTMLResponse(UI_HTML)

if __name__ == "__main__":
    if is_kag:
        def run_uvicorn():
            config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="error")
            server = uvicorn.Server(config)
            server.run()
            
        print("\n[START] SERVER RUNNING IN BACKGROUND...")
        threading.Thread(target=run_uvicorn, daemon=True).start()
        try:
            while True: time.sleep(100)
        except KeyboardInterrupt: pass
    else:
        print("\n[STRICT] KAGGLE ONLY.")
