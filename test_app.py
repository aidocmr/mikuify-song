from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

import app


def test_find_vocal_segments_skips_long_silence_and_merges_brief_gaps() -> None:
    sr = app.SAMPLE_RATE
    # Pure silence -> no segments
    assert app.find_vocal_segments(np.zeros(sr * 2, dtype=np.float32), sr) == []

    # 6s track: active [1.0, 1.5], 0.3s breath (merged), active [1.8, 2.3], 2.0s silence (skipped), active [4.3, 4.8]
    y = np.zeros(sr * 6, dtype=np.float32)
    y[int(1.0 * sr):int(1.5 * sr)] = 0.5
    y[int(1.8 * sr):int(2.3 * sr)] = 0.5
    y[int(4.3 * sr):int(4.8 * sr)] = 0.5

    segs = app.find_vocal_segments(y, sr, pad_s=0.25, min_silence_s=0.8)
    assert len(segs) == 2, f"Expected 2 segments, got {segs}"
    # First merged span covers [0.75s, 2.55s], second covers [4.05s, 5.05s]
    assert abs(segs[0][0] - int(0.75 * sr)) <= sr // 10
    assert abs(segs[0][1] - int(2.55 * sr)) <= sr // 10
    assert abs(segs[1][0] - int(4.05 * sr)) <= sr // 10
    assert abs(segs[1][1] - int(5.05 * sr)) <= sr // 10


def test_fit_segment_and_mix_cover_prevent_clipping() -> None:
    sr = app.SAMPLE_RATE
    # 40 kHz RVC output slightly shorter than 1.0s at 44.1 kHz
    miku_40k = np.full(39500, 26000, dtype=np.int16)
    seg = app._fit_segment(miku_40k, 40000, sr, sr, fade_in=True, fade_out=True)
    assert seg.shape == (sr,) and seg.dtype == np.float32
    assert abs(seg[0]) < 1e-4 and abs(seg[-1]) < 1e-4  # edges smoothly faded to 0

    instrumental = np.full((2, sr), 0.7, dtype=np.float32)
    cover, miku_st = app.mix_cover(seg, instrumental, vocal_gain_db=3.0)
    assert cover.shape == (sr, 2) and cover.dtype == np.int16
    assert miku_st.shape == (sr, 2) and miku_st.dtype == np.int16
    assert int(np.abs(cover).max()) <= int(0.981 * 32767)


def test_decode_to_wav_rejects_corrupt_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.mp3"
        bad.write_bytes(b"not audio")
        try:
            app.decode_to_wav(bad, Path(tmp) / "out.wav")
        except gr.Error:
            return
        raise AssertionError("Expected gr.Error for invalid audio input")


def test_mikuify_song_end_to_end_and_silent_skip() -> None:
    sr = 22050
    with tempfile.TemporaryDirectory() as tmp:
        # 1) Completely silent track -> skips RVC entirely, returns zeros cleanly
        silent_wav = Path(tmp) / "silent.wav"
        sf.write(silent_wav, np.zeros(sr, dtype=np.float32), sr)
        (s_sr, s_cover), (sv_sr, s_vocals), _ = app.mikuify_song(str(silent_wav))
        assert s_sr == sv_sr == app.SAMPLE_RATE
        assert np.abs(s_vocals).max() == 0

        # 2) Track with leading silence + vocal-like harmonic audio -> exercises separation + silence skip + RVC + remix
        t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
        f0 = 330.0 + 15.0 * np.sin(2 * np.pi * 5.5 * t)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        active = sum((0.25 / k) * np.sin(k * phase) for k in range(1, 8)).astype(np.float32)
        sig = np.concatenate([np.zeros(sr * 2, dtype=np.float32), active])
        in_wav = Path(tmp) / "lead_silence.wav"
        sf.write(in_wav, sig, sr)
        (c_sr, cover), (v_sr, vocals), (i_sr, inst) = app.mikuify_song(
            str(in_wav), pitch_shift=0, index_rate=0.5, vocal_gain_db=0.0
        )

    assert c_sr == v_sr == i_sr == app.SAMPLE_RATE
    assert cover.shape == vocals.shape == inst.shape
    assert cover.ndim == 2 and cover.shape[1] == 2
    assert cover.dtype == vocals.dtype == inst.dtype == np.int16
    # First 1.5s of pure leading silence stays strictly silent; active tail is converted by Miku RVC
    assert np.abs(vocals[: int(app.SAMPLE_RATE * 1.5)]).max() == 0
    assert np.abs(vocals[int(app.SAMPLE_RATE * 2.2) :]).max() > 0


if __name__ == "__main__":
    test_find_vocal_segments_skips_long_silence_and_merges_brief_gaps()
    test_fit_segment_and_mix_cover_prevent_clipping()
    test_decode_to_wav_rejects_corrupt_file()
    test_mikuify_song_end_to_end_and_silent_skip()
    print("ALL TESTS PASSED")
