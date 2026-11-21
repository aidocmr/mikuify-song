from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import soxr
import torch
from demucs_onnx import separate_stem
from huggingface_hub import hf_hub_download
from infer_rvc_python import BaseLoader

SAMPLE_RATE = 44100
MIKU_REPO = "NoCrypt/miku_RVC"
MIKU_PTH = "1a_miku_default_rvc_(aple)/miku_default_rvc.pth"
MIKU_INDEX = "1a_miku_default_rvc_(aple)/added_IVF4457_Flat_nprobe_1_miku_default_rvc_v2.index"
RMVPE_REPO = "r3gm/sonitranslate_voice_models"
RMVPE_FILE = "rmvpe.pt"

_rvc_lock = threading.Lock()
_rvc_state: tuple[BaseLoader, str, str] | None = None


def _get_rvc() -> tuple[BaseLoader, str, str]:
    global _rvc_state
    if _rvc_state is None:
        pth = hf_hub_download(MIKU_REPO, MIKU_PTH)
        idx = hf_hub_download(MIKU_REPO, MIKU_INDEX)
        rmvpe = hf_hub_download(RMVPE_REPO, RMVPE_FILE)
        loader = BaseLoader(only_cpu=not torch.cuda.is_available(), rmvpe_path=rmvpe)
        _rvc_state = (loader, pth, idx)
    return _rvc_state


def decode_to_wav(src: str | Path, dst: Path) -> None:
    """Decode any audio/video file to 44.1 kHz stereo float32 WAV for StemSplitio."""
    if not shutil.which("ffmpeg"):
        raise gr.Error("ffmpeg is required to decode audio files.")
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn", "-ar", str(SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_f32le", str(dst)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise gr.Error("Could not decode the uploaded audio file. Please upload a valid song.")


def find_vocal_segments(
    mono: np.ndarray,
    sr: int = SAMPLE_RATE,
    top_db: float = 40.0,
    abs_floor: float = 1e-3,
    frame_ms: int = 50,
    pad_s: float = 0.25,
    min_silence_s: float = 0.8,
) -> list[tuple[int, int]]:
    """Return merged non-silent (start, end) sample spans so RVC skips instrumental gaps."""
    n = len(mono)
    peak = float(np.abs(mono).max()) if n else 0.0
    if peak < abs_floor:
        return []

    thresh = max(abs_floor, peak * (10.0 ** (-top_db / 20.0)))
    frame = max(1, int(sr * frame_ms / 1000))
    pad_rem = (-n) % frame
    frames = np.pad(mono, (0, pad_rem)).reshape(-1, frame)
    active = np.flatnonzero(np.sqrt(np.mean(frames * frames, axis=1)) > thresh)
    if active.size == 0:
        return []

    min_gap = int(sr * min_silence_s)
    raw: list[tuple[int, int]] = []
    for idx in active:
        s, e = int(idx) * frame, min(n, (int(idx) + 1) * frame)
        if raw and s - raw[-1][1] < min_gap:
            raw[-1] = (raw[-1][0], e)
        else:
            raw.append((s, e))

    pad = int(sr * pad_s)
    merged: list[tuple[int, int]] = []
    for s, e in raw:
        ps, pe = max(0, s - pad), min(n, e + pad)
        if merged and ps <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], pe))
        else:
            merged.append((ps, pe))
    return merged


def _fit_segment(
    audio_int16: np.ndarray,
    src_sr: int,
    target_sr: int,
    target_len: int,
    fade_in: bool,
    fade_out: bool,
) -> np.ndarray:
    """Resample an RVC int16 segment to target_sr, align to target_len, and de-click edges."""
    seg = audio_int16.astype(np.float32) / 32768.0
    if src_sr != target_sr:
        seg = soxr.resample(seg, src_sr, target_sr, quality="HQ").astype(np.float32)
    if len(seg) < target_len:
        seg = np.pad(seg, (0, target_len - len(seg)))
    elif len(seg) > target_len:
        seg = seg[:target_len]

    fade = min(int(target_sr * 0.015), target_len // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if fade_in:
            seg[:fade] *= ramp
        if fade_out:
            seg[-fade:] *= ramp[::-1]
    return seg


def _to_int16_stereo(stereo_ch_first: np.ndarray) -> np.ndarray:
    peak = float(np.abs(stereo_ch_first).max())
    if peak > 0.98:
        stereo_ch_first = stereo_ch_first * (0.98 / peak)
    return np.clip(stereo_ch_first.T * 32767.0, -32768, 32767).astype(np.int16)


def mix_cover(
    miku_mono: np.ndarray,
    instrumental: np.ndarray,
    vocal_gain_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply gain and mix aligned float32 mono Miku vocals with stereo instrumental."""
    miku = miku_mono * float(10.0 ** (vocal_gain_db / 20.0))
    miku_stereo = np.stack([miku, miku], axis=0)
    return _to_int16_stereo(instrumental + miku_stereo), _to_int16_stereo(miku_stereo)


def mikuify_song(
    audio_path: str | None,
    pitch_shift: int = 0,
    index_rate: float = 0.66,
    vocal_gain_db: float = 0.0,
    progress: gr.Progress = gr.Progress(),
):
    if not audio_path or not Path(audio_path).is_file():
        raise gr.Error("Please upload a song first.")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "input.wav"
        progress(0.1, desc="Decoding audio...")
        decode_to_wav(audio_path, wav_path)

        mix, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        mix = mix.T  # (2, samples)
        if mix.shape[1] < SAMPLE_RATE // 4:
            raise gr.Error("Audio clip is too short (minimum 0.25 seconds).")

        progress(0.25, desc="Separating vocals with StemSplitio HT-Demucs ONNX (CPU)...")
        # fp16weights halves download size (~166 MB) with <6e-5 diff vs fp32; CPU EP is faster than CUDA here
        vocals = separate_stem(wav_path, "vocals", precision="fp16weights", providers="cpu", progress=False)
        instrumental = mix - vocals

        vocal_mono = vocals.mean(axis=0)
        segments = find_vocal_segments(vocal_mono, sr)
        miku_full = np.zeros(len(vocal_mono), dtype=np.float32)

        if segments:
            active_samples = sum(e - s for s, e in segments)
            skipped_pct = int(round(100.0 * (1.0 - active_samples / len(vocal_mono))))
            # shortcut: single process lock serializes GPU/CPU RVC calls; switch to a worker pool if serving high concurrency
            with _rvc_lock:
                loader, pth, idx = _get_rvc()
                loader.apply_conf(
                    tag="miku",
                    file_model=pth,
                    file_index=idx,
                    pitch_algo="rmvpe+",
                    pitch_lvl=int(pitch_shift),
                    index_influence=float(index_rate),
                )
                for i, (start, end) in enumerate(segments):
                    progress(
                        0.6 + 0.3 * (i / len(segments)),
                        desc=f"Converting vocals ({i + 1}/{len(segments)}, skipped {skipped_pct}% silence)...",
                    )
                    seg_raw, seg_sr = loader.generate_from_cache((vocal_mono[start:end], sr), tag="miku")
                    miku_full[start:end] = _fit_segment(
                        seg_raw, seg_sr, sr, end - start,
                        fade_in=start > 0,
                        fade_out=end < len(vocal_mono),
                    )

        progress(0.92, desc="Mixing cover...")
        cover_i16, miku_i16 = mix_cover(miku_full, instrumental, vocal_gain_db)
        inst_i16 = _to_int16_stereo(instrumental)
        progress(1.0, desc="Done!")

    return (sr, cover_i16), (sr, miku_i16), (sr, inst_i16)


with gr.Blocks(title="Mikuify Song") as demo:
    gr.Markdown(
        "# Mikuify Song\n"
        "Turn any song into a **Hatsune Miku** cover using "
        "[StemSplitio HT-Demucs ONNX](https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx) "
        "and [NoCrypt Miku RVC (`1a_miku_default_rvc_(aple)`)](https://huggingface.co/NoCrypt/miku_RVC/tree/main/1a_miku_default_rvc_(aple))."
    )
    with gr.Row():
        with gr.Column():
            song_in = gr.Audio(label="Input Song", type="filepath")
            pitch = gr.Slider(
                -12, 12, value=0, step=1,
                label="Pitch Shift (semitones)",
                info="Tip: use +12 for male vocals, 0 for female vocals.",
            )
            index_rate = gr.Slider(0.0, 1.0, value=0.66, step=0.01, label="Miku Voice Index Strength")
            vocal_gain = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="Vocal Gain (dB)")
            run_btn = gr.Button("Mikuify!", variant="primary")
        with gr.Column():
            cover_out = gr.Audio(label="Hatsune Miku Cover", interactive=False)
            with gr.Accordion("Separated Stems", open=False):
                miku_vocals_out = gr.Audio(label="Miku Vocals", interactive=False)
                inst_out = gr.Audio(label="Instrumental", interactive=False)

    run_btn.click(
        fn=mikuify_song,
        inputs=[song_in, pitch, index_rate, vocal_gain],
        outputs=[cover_out, miku_vocals_out, inst_out],
    )


if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft(primary_hue="teal"))
