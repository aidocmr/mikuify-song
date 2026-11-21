---
title: Mikuify Song
emoji: 🎤
colorFrom: teal
colorTo: cyan
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
suggested_hardware: t4-small
short_description: Turn any song into a Hatsune Miku cover
---

# Mikuify Song

Gradio Space app that turns any song into a Hatsune Miku cover by:
1. Separating vocals with [StemSplitio/htdemucs-ft-vocals-onnx](https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx) (`demucs-onnx`).
2. Converting vocals with [NoCrypt/miku_RVC (`1a_miku_default_rvc_(aple)`)](https://huggingface.co/NoCrypt/miku_RVC/tree/main/1a_miku_default_rvc_(aple)) (`infer-rvc-python`).
3. Remixing the Miku vocals back onto the instrumental.

## Run locally with `uv`

Requires `ffmpeg` on `PATH`.

```bash
uv sync
uv run python app.py
```

## Deploy to Hugging Face Spaces

Push this repository to a Gradio Space. `README.md`, `requirements.txt`, `packages.txt`, and `app.py` are preconfigured; models download automatically from Hugging Face Hub on first run.
