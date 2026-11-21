from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

import app


def test_find_vocal_segments_skips_long_silence_and_merges_brief_gaps() -> None:
    sr = app.SAMPLE_RATE
    assert app.find_vocal_segments(np.zeros(sr * 2, dtype=np.float32), sr) == []

    # 6s track: active [1.0, 1.5], 0.3s breath (merged), active [1.8, 2.3], 2.0s silence (skipped), active [4.3, 4.8]
    y = np.zeros(sr * 6, dtype=np.float32)
    y[int(1.0 * sr) : int(1.5 * sr)] = 0.5
    y[int(1.8 * sr) : int(2.3 * sr)] = 0.5
    y[int(4.3 * sr) : int(4.8 * sr)] = 0.5

    segs = app.find_vocal_segments(y, sr, pad_s=0.25, min_silence_s=0.8)
    assert len(segs) == 2, f"Expected 2 segments, got {segs}"
    assert abs(segs[0][0] - int(0.75 * sr)) <= sr // 10
    assert abs(segs[0][1] - int(2.55 * sr)) <= sr // 10
    assert abs(segs[1][0] - int(4.05 * sr)) <= sr // 10
    assert abs(segs[1][1] - int(5.05 * sr)) <= sr // 10


def test_fit_segment_and_mix_cover_prevent_clipping() -> None:
    sr = app.SAMPLE_RATE
    miku_40k = np.full(39500, 26000, dtype=np.int16)
    seg = app._fit_segment(miku_40k, 40000, sr, sr, fade_in=True, fade_out=True)
    assert seg.shape == (sr,) and seg.dtype == np.float32
    assert abs(seg[0]) < 1e-4 and abs(seg[-1]) < 1e-4

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


def test_separate_and_convert_two_stage_both_modes() -> None:
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
    f0 = 330.0 + 15.0 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    active = sum((0.25 / k) * np.sin(k * phase) for k in range(1, 8)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        sig = np.concatenate([np.zeros(sr * 2, dtype=np.float32), active])
        in_wav = tmp_dir / "lead_silence.wav"
        sf.write(in_wav, sig, sr)

        # Stage 1 (Combined): separate stems without running RVC
        (ol_sr, orig_lead), orig_backing, (i_sr, inst) = app.separate_song(str(in_wav), app.MODE_COMBINED)
        assert ol_sr == i_sr == app.SAMPLE_RATE
        assert orig_backing is None
        assert orig_lead.shape == inst.shape and orig_lead.shape[1] == 2

        # Save separated stems as Gradio's Audio(type="filepath") does, then run Stage 2
        lead_wav = tmp_dir / "orig_lead.wav"
        inst_wav = tmp_dir / "inst.wav"
        sf.write(lead_wav, orig_lead, ol_sr)
        sf.write(inst_wav, inst, i_sr)

        (c_sr, cover), (v_sr, miku_v), miku_b = app.convert_stems_to_miku(
            str(lead_wav), None, str(inst_wav),
            voice_name="Hatsune Miku (Default)", pitch_shift=0, index_rate=0.5, vocal_gain_db=0.0,
        )
        assert c_sr == v_sr == app.SAMPLE_RATE
        assert miku_b is None
        assert cover.shape == miku_v.shape == inst.shape
        assert np.abs(miku_v[: int(app.SAMPLE_RATE * 1.5)]).max() == 0
        assert np.abs(miku_v[int(app.SAMPLE_RATE * 2.2) :]).max() > 0

        # Verify GUMI (zipped 48kHz RVC v2 model) works cleanly too
        (g_sr, g_cover), (gv_sr, g_vocals), g_back = app.convert_stems_to_miku(
            str(lead_wav), None, str(inst_wav),
            voice_name="GUMI (Megpoid)", pitch_shift=0, index_rate=0.5, vocal_gain_db=0.0,
        )
        assert g_sr == gv_sr == app.SAMPLE_RATE and g_back is None
        assert g_cover.shape == g_vocals.shape == inst.shape
        assert np.abs(g_vocals[int(app.SAMPLE_RATE * 2.2) :]).max() > 0

        # Stage 1 + 2 (Lead + Backing): Miku lead + Kasane Teto backing with separate params
        (ll_sr, lead_st), backing_st_tuple, (li_sr, inst2) = app.separate_song(str(in_wav), app.MODE_LEAD_BACKING)
        assert ll_sr == li_sr == app.SAMPLE_RATE and backing_st_tuple is not None
        lb_sr, backing_st = backing_st_tuple
        assert lb_sr == app.SAMPLE_RATE and lead_st.shape == backing_st.shape == inst2.shape

        lead2_wav = tmp_dir / "lead2.wav"
        back2_wav = tmp_dir / "back2.wav"
        inst2_wav = tmp_dir / "inst2.wav"
        sf.write(lead2_wav, lead_st, ll_sr)
        sf.write(back2_wav, backing_st, lb_sr)
        sf.write(inst2_wav, inst2, li_sr)

        (c2_sr, cover2), (ml_sr, miku_l), miku_b_tuple = app.convert_stems_to_miku(
            str(lead2_wav), str(back2_wav), str(inst2_wav),
            voice_name="Hatsune Miku (Default)", pitch_shift=0, index_rate=0.5, vocal_gain_db=0.0,
            backing_voice_name="Kasane Teto (UTAU)", backing_pitch_shift=2, backing_index_rate=0.4, backing_gain_db=-3.0,
        )
        assert c2_sr == ml_sr == app.SAMPLE_RATE and miku_b_tuple is not None
        mb_sr, miku_b2 = miku_b_tuple
        assert mb_sr == app.SAMPLE_RATE and cover2.shape == miku_l.shape == miku_b2.shape == inst2.shape
        assert np.abs(miku_l[int(app.SAMPLE_RATE * 2.2) :]).max() > 0


if __name__ == "__main__":
    test_find_vocal_segments_skips_long_silence_and_merges_brief_gaps()
    test_fit_segment_and_mix_cover_prevent_clipping()
    test_decode_to_wav_rejects_corrupt_file()
    test_separate_and_convert_two_stage_both_modes()
    print("ALL TESTS PASSED")
