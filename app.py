from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import soxr
import torch
from audio_separator.separator import Separator
from demucs_onnx import separate_stem
from huggingface_hub import hf_hub_download
from infer_rvc_python import BaseLoader

SAMPLE_RATE = 44100
RMVPE_REPO = "r3gm/sonitranslate_voice_models"
RMVPE_FILE = "rmvpe.pt"
KARAOKE_MODEL = "UVR_MDXNET_KARA_2.onnx"
KARAOKE_CACHE_DIR = Path.home() / ".cache" / "audio-separator-models"
VOICE_CACHE_DIR = Path.home() / ".cache" / "mikuify-voices"

VOICES: dict[str, dict[str, str]] = {
    "Hatsune Miku (Default)": {
        "repo": "NoCrypt/miku_RVC",
        "pth": "1a_miku_default_rvc_(aple)/miku_default_rvc.pth",
        "index": "1a_miku_default_rvc_(aple)/added_IVF4457_Flat_nprobe_1_miku_default_rvc_v2.index",
    },
    "Kasane Teto (UTAU)": {
        "repo": "SinsuSquid/Kasane-Teto-UTAU-RVC-v2",
        "pth": "KasaneTeto.pth",
        "index": "KasaneTeto.index",
    },
    "GUMI (Megpoid)": {
        "repo": "EpiphanyMoon12/vocaloid",
        "zip": "GumiRVC.zip",
        "pth": "GumiRVC.pth",
        "index": "added_IVF1117_Flat_nprobe_1_GumiRVC_v2.index",
    },
}
DEFAULT_VOICE = "Hatsune Miku (Default)"

MODE_COMBINED = "StemSplitio HT-Demucs (Combined Vocals)"
MODE_LEAD_BACKING = "StemSplitio + UVR MDX-Net Karaoke 2 (Lead + Backing Vocals)"

_rvc_lock = threading.Lock()
_rvc_loader: BaseLoader | None = None
_voice_paths: dict[str, tuple[str, str]] = {}


def _get_rvc_loader() -> BaseLoader:
    global _rvc_loader
    if _rvc_loader is None:
        rmvpe = hf_hub_download(RMVPE_REPO, RMVPE_FILE)
        _rvc_loader = BaseLoader(only_cpu=not torch.cuda.is_available(), rmvpe_path=rmvpe)
    return _rvc_loader


def resolve_voice(voice_name: str) -> tuple[str, str]:
    """Download (and extract if zipped) the selected RVC voice checkpoint and index."""
    if voice_name not in VOICES:
        raise gr.Error(f"Unknown voice model: {voice_name}")
    if voice_name in _voice_paths:
        return _voice_paths[voice_name]

    spec = VOICES[voice_name]
    if "zip" in spec:
        zip_path = hf_hub_download(spec["repo"], spec["zip"])
        out_dir = VOICE_CACHE_DIR / Path(spec["zip"]).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        pth_path = out_dir / Path(spec["pth"]).name
        idx_path = out_dir / Path(spec["index"]).name
        if not (pth_path.is_file() and idx_path.is_file()):
            with zipfile.ZipFile(zip_path) as zf:
                for member, dest in ((spec["pth"], pth_path), (spec["index"], idx_path)):
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        paths = (str(pth_path), str(idx_path))
    else:
        paths = (
            hf_hub_download(spec["repo"], spec["pth"]),
            hf_hub_download(spec["repo"], spec["index"]),
        )
    _voice_paths[voice_name] = paths
    return paths


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


def _align_stereo(stem: np.ndarray, n: int) -> np.ndarray:
    if stem.shape[1] < n:
        return np.pad(stem, ((0, 0), (0, n - stem.shape[1])))
    return stem[:, :n]


def separate_lead_backing(vocals: np.ndarray, sr: int, work_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Split an isolated stereo vocals stem into (lead, backing) using UVR MDX-Net Karaoke 2."""
    n = vocals.shape[1]
    if not find_vocal_segments(vocals.mean(axis=0), sr):
        zeros = np.zeros_like(vocals)
        return zeros, zeros.copy()

    vocals_wav = work_dir / "all_vocals.wav"
    sf.write(vocals_wav, vocals.T, sr, subtype="FLOAT")

    KARAOKE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sep = Separator(
        log_level=logging.ERROR,
        output_dir=str(work_dir),
        model_file_dir=str(KARAOKE_CACHE_DIR),
        normalization_threshold=1.0,
    )
    sep.load_model(KARAOKE_MODEL)
    out_files = sep.separate(str(vocals_wav))

    lead = backing = None
    for name in out_files:
        data, _ = sf.read(work_dir / name, dtype="float32", always_2d=True)
        stereo = _align_stereo(data.T, n)
        if "(Vocals)" in name:
            lead = stereo
        elif "(Instrumental)" in name:
            backing = stereo

    if lead is None or backing is None:
        raise gr.Error("Lead/backing vocal separation failed to produce both stems.")
    return lead, backing


def _convert_vocal_stem(
    vocal_mono: np.ndarray,
    sr: int,
    voice_name: str,
    pitch_shift: int,
    index_rate: float,
    progress: gr.Progress,
    prog_start: float,
    prog_end: float,
    label: str,
) -> np.ndarray:
    """Convert non-silent spans of a mono vocal stem using the selected RVC voice."""
    segments = find_vocal_segments(vocal_mono, sr)
    out = np.zeros(len(vocal_mono), dtype=np.float32)
    if not segments:
        return out

    active_samples = sum(e - s for s, e in segments)
    skipped_pct = int(round(100.0 * (1.0 - active_samples / len(vocal_mono))))
    # shortcut: single process lock serializes GPU/CPU RVC calls; switch to a worker pool if serving high concurrency
    with _rvc_lock:
        pth, idx = resolve_voice(voice_name)
        loader = _get_rvc_loader()
        loader.apply_conf(
            tag=voice_name,
            file_model=pth,
            file_index=idx,
            pitch_algo="rmvpe+",
            pitch_lvl=int(pitch_shift),
            index_influence=float(index_rate),
        )
        for i, (start, end) in enumerate(segments):
            frac = i / len(segments)
            progress(
                prog_start + (prog_end - prog_start) * frac,
                desc=f"Converting {label} with {voice_name} ({i + 1}/{len(segments)}, skipped {skipped_pct}% silence)...",
            )
            seg_raw, seg_sr = loader.generate_from_cache((vocal_mono[start:end], sr), tag=voice_name)
            out[start:end] = _fit_segment(
                seg_raw, seg_sr, sr, end - start,
                fade_in=start > 0,
                fade_out=end < len(vocal_mono),
            )
    return out


def _to_int16_stereo(stereo_ch_first: np.ndarray) -> np.ndarray:
    peak = float(np.abs(stereo_ch_first).max())
    if peak > 0.98:
        stereo_ch_first = stereo_ch_first * (0.98 / peak)
    return np.clip(stereo_ch_first.T * 32767.0, -32768, 32767).astype(np.int16)


def _read_stereo(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    stereo = data.T
    if stereo.shape[0] == 1:
        stereo = np.tile(stereo, (2, 1))
    elif stereo.shape[0] > 2:
        stereo = stereo[:2]
    if sr != target_sr:
        stereo = soxr.resample(stereo.T, sr, target_sr, quality="HQ").T.astype(np.float32)
    return stereo


def mix_cover(
    miku_mono: np.ndarray,
    instrumental: np.ndarray,
    vocal_gain_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply gain and mix aligned float32 mono Miku vocals with stereo instrumental."""
    miku = miku_mono * float(10.0 ** (vocal_gain_db / 20.0))
    miku_stereo = np.stack([miku, miku], axis=0)
    return _to_int16_stereo(instrumental + miku_stereo), _to_int16_stereo(miku_stereo)


def separate_song(
    audio_path: str | None,
    sep_mode: str = MODE_COMBINED,
    progress: gr.Progress = gr.Progress(),
):
    """Separate a song into original stems: (lead_or_vocals, backing_or_none, instrumental)."""
    if not audio_path or not Path(audio_path).is_file():
        raise gr.Error("Please upload a song first.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        wav_path = tmp_dir / "input.wav"
        progress(0.1, desc="Decoding audio...")
        decode_to_wav(audio_path, wav_path)

        mix, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        mix = mix.T  # (2, samples)
        if mix.shape[1] < SAMPLE_RATE // 4:
            raise gr.Error("Audio clip is too short (minimum 0.25 seconds).")

        progress(0.3, desc="Separating vocals with StemSplitio HT-Demucs ONNX (CPU)...")
        # fp16weights halves download size (~166 MB) with <6e-5 diff vs fp32; CPU EP is faster than CUDA here
        vocals = separate_stem(wav_path, "vocals", precision="fp16weights", providers="cpu", progress=False)
        instrumental = mix - vocals

        if sep_mode == MODE_LEAD_BACKING:
            progress(0.7, desc="Splitting lead & backing vocals with UVR MDX-Net Karaoke 2...")
            lead, backing = separate_lead_backing(vocals, sr, tmp_dir)
            orig_lead = (sr, _to_int16_stereo(lead))
            orig_backing = (sr, _to_int16_stereo(backing))
        else:
            orig_lead = (sr, _to_int16_stereo(vocals))
            orig_backing = None

        inst_out = (sr, _to_int16_stereo(instrumental))
        progress(1.0, desc="Stems ready!")

    return orig_lead, orig_backing, inst_out


def convert_stems_to_miku(
    orig_lead_path: str | None,
    orig_backing_path: str | None,
    inst_path: str | None,
    voice_name: str = DEFAULT_VOICE,
    pitch_shift: int = 0,
    index_rate: float = 0.66,
    vocal_gain_db: float = 0.0,
    backing_voice_name: str = DEFAULT_VOICE,
    backing_pitch_shift: int = 0,
    backing_index_rate: float = 0.66,
    backing_gain_db: float = 0.0,
    progress: gr.Progress = gr.Progress(),
):
    """Convert separated original vocal stems with the chosen RVC voice(s) and remix."""
    if not orig_lead_path or not Path(orig_lead_path).is_file():
        raise gr.Error("Please separate stems first (missing original vocals).")
    if not inst_path or not Path(inst_path).is_file():
        raise gr.Error("Please separate stems first (missing instrumental).")

    progress(0.05, desc="Loading original stems...")
    instrumental = _read_stereo(inst_path, SAMPLE_RATE)
    n = instrumental.shape[1]
    lead = _align_stereo(_read_stereo(orig_lead_path, SAMPLE_RATE), n)
    has_backing = bool(orig_backing_path and Path(orig_backing_path).is_file())

    if has_backing:
        backing = _align_stereo(_read_stereo(orig_backing_path, SAMPLE_RATE), n)
        miku_lead = _convert_vocal_stem(
            lead.mean(axis=0), SAMPLE_RATE, voice_name, pitch_shift, index_rate, progress, 0.1, 0.55, "lead vocals"
        )
        miku_backing = _convert_vocal_stem(
            backing.mean(axis=0), SAMPLE_RATE, backing_voice_name, backing_pitch_shift, backing_index_rate,
            progress, 0.55, 0.9, "backing vocals",
        )
        lead_scaled = miku_lead * float(10.0 ** (vocal_gain_db / 20.0))
        backing_scaled = miku_backing * float(10.0 ** (backing_gain_db / 20.0))
        cover_i16, _ = mix_cover(lead_scaled + backing_scaled, instrumental, 0.0)
        lead_i16 = _to_int16_stereo(np.stack([lead_scaled, lead_scaled], axis=0))
        backing_out = (SAMPLE_RATE, _to_int16_stereo(np.stack([backing_scaled, backing_scaled], axis=0)))
    else:
        miku_all = _convert_vocal_stem(
            lead.mean(axis=0), SAMPLE_RATE, voice_name, pitch_shift, index_rate, progress, 0.1, 0.9, "vocals"
        )
        cover_i16, lead_i16 = mix_cover(miku_all, instrumental, vocal_gain_db)
        backing_out = None

    progress(1.0, desc="Done!")
    return (SAMPLE_RATE, cover_i16), (SAMPLE_RATE, lead_i16), backing_out


with gr.Blocks(title="Mikuify Song") as demo:
    gr.Markdown(
        "# Mikuify Song\n"
        "Turn any song into a **Hatsune Miku**, **Kasane Teto**, or **GUMI** cover using "
        "[StemSplitio HT-Demucs ONNX](https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx), "
        "optional `UVR_MDXNET_KARA_2` lead/backing vocal splitting, and RVC v2."
    )
    with gr.Row():
        with gr.Column():
            song_in = gr.Audio(label="Input Song", type="filepath")
            sep_mode = gr.Radio(
                choices=[MODE_COMBINED, MODE_LEAD_BACKING],
                value=MODE_COMBINED,
                label="Vocal Separation Model",
                info="Lead + Backing splits harmonies/backing vocals and lets you tune them separately.",
            )
            with gr.Row():
                sep_only_btn = gr.Button("Separate Stems Only", variant="secondary")
                run_btn = gr.Button("Separate & Mikuify!", variant="primary")
            voice_select = gr.Dropdown(
                choices=list(VOICES),
                value=DEFAULT_VOICE,
                label="Lead / Vocal Voice Model",
            )
            pitch = gr.Slider(
                -12, 12, value=0, step=1,
                label="Lead / Vocal Pitch Shift (semitones)",
                info="Tip: +12 for male vocals, 0 for female vocals, or -2/-3 on Miku for classic Akita Neru.",
            )
            index_rate = gr.Slider(0.0, 1.0, value=0.66, step=0.01, label="Lead / Vocal Index Strength")
            vocal_gain = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="Lead / Vocal Gain (dB)")
            with gr.Group(visible=False) as backing_controls:
                gr.Markdown("### Backing Vocals Settings")
                backing_voice_select = gr.Dropdown(
                    choices=list(VOICES),
                    value=DEFAULT_VOICE,
                    label="Backing Voice Model",
                )
                backing_pitch = gr.Slider(-12, 12, value=0, step=1, label="Backing Pitch Shift (semitones)")
                backing_index_rate = gr.Slider(0.0, 1.0, value=0.66, step=0.01, label="Backing Index Strength")
                backing_gain = gr.Slider(-12.0, 12.0, value=0.0, step=0.5, label="Backing Vocal Gain (dB)")
        with gr.Column():
            with gr.Group():
                gr.Markdown("### Original Stems")
                orig_lead_out = gr.Audio(label="Original Vocals / Lead", type="filepath")
                orig_backing_out = gr.Audio(label="Original Backing Vocals", type="filepath", visible=False)
                inst_out = gr.Audio(label="Instrumental", type="filepath")
                convert_btn = gr.Button("Convert Original Stems", variant="primary")
            cover_out = gr.Audio(label="Converted Cover", interactive=False)
            with gr.Accordion("Converted Vocal Stems", open=False):
                miku_lead_out = gr.Audio(label="Converted Vocals / Lead", interactive=False)
                miku_backing_out = gr.Audio(label="Converted Backing Vocals", interactive=False)

    sep_mode.change(
        fn=lambda mode: (gr.update(visible=mode == MODE_LEAD_BACKING), gr.update(visible=mode == MODE_LEAD_BACKING)),
        inputs=[sep_mode],
        outputs=[backing_controls, orig_backing_out],
    )
    miku_inputs = [
        orig_lead_out, orig_backing_out, inst_out,
        voice_select, pitch, index_rate, vocal_gain,
        backing_voice_select, backing_pitch, backing_index_rate, backing_gain,
    ]
    miku_outputs = [cover_out, miku_lead_out, miku_backing_out]

    sep_only_btn.click(
        fn=separate_song,
        inputs=[song_in, sep_mode],
        outputs=[orig_lead_out, orig_backing_out, inst_out],
    )
    convert_btn.click(
        fn=convert_stems_to_miku,
        inputs=miku_inputs,
        outputs=miku_outputs,
    )
    # Full pipeline: show original stems first, then immediately run RVC conversion on them
    run_btn.click(
        fn=separate_song,
        inputs=[song_in, sep_mode],
        outputs=[orig_lead_out, orig_backing_out, inst_out],
    ).then(
        fn=convert_stems_to_miku,
        inputs=miku_inputs,
        outputs=miku_outputs,
    )


if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft(primary_hue="teal"))
