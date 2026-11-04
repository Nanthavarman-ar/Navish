#!/usr/bin/env bash
# One-shot setup for a Vast.ai GPU instance (Ubuntu + CUDA PyTorch image).
# Installs: ComfyUI + Wan 2.2 (I2V 14B, S2V lip-sync) + RIFE + audio pipeline + musubi-tuner (LoRA training)
#
#   bash vast_setup.sh            # everything
#   SKIP_S2V=1 bash vast_setup.sh # skip the lip-sync model (~16 GB less)
#   HF_TOKEN=hf_xxx bash vast_setup.sh   # faster / rate-limit-free downloads
set -euo pipefail
W=${WORKSPACE:-/workspace}
cd "$W"

echo "== system packages"
apt-get update -qq && apt-get install -y -qq ffmpeg git git-lfs aria2 fonts-dejavu > /dev/null

echo "== ComfyUI"
[ -d ComfyUI ] || git clone -q https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -q -r requirements.txt
cd custom_nodes
[ -d ComfyUI-Manager ]           || git clone -q https://github.com/ltdrdata/ComfyUI-Manager.git
[ -d ComfyUI-VideoHelperSuite ]  || git clone -q https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
[ -d ComfyUI-Frame-Interpolation ] || git clone -q https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git
for d in ComfyUI-VideoHelperSuite ComfyUI-Frame-Interpolation; do
  [ -f "$d/requirements.txt" ] && pip install -q -r "$d/requirements.txt" || true
done
cd ..

echo "== Wan 2.2 models"
pip install -q -U "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1
get () {  # repo  path-in-repo  target-folder
  local f; f=$(basename "$2")
  if [ -f "models/$3/$f" ]; then echo "   have $f"; return; fi
  hf download "$1" "$2" --local-dir /tmp/hfdl >/dev/null
  mkdir -p "models/$3" && mv "/tmp/hfdl/$2" "models/$3/$f" && echo "   got $f"
}
R22=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
R21=Comfy-Org/Wan_2.1_ComfyUI_repackaged
get $R22 split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors diffusion_models
get $R22 split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors  diffusion_models
get $R21 split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors              text_encoders
get $R22 split_files/vae/wan_2.1_vae.safetensors                                       vae
if [ -z "${SKIP_S2V:-}" ]; then
  get $R22 split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors          diffusion_models
  get $R22 split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors          audio_encoders
fi
cd "$W"

echo "== audio pipeline (Chatterbox TTS + faster-whisper)"
pip install -q chatterbox-tts faster-whisper

echo "== musubi-tuner (Wan 2.2 LoRA training)"
[ -d musubi-tuner ] || git clone -q https://github.com/kohya-ss/musubi-tuner.git
pip install -q -e musubi-tuner || echo "   musubi-tuner install needs a look - see its README"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cat <<EOF

DONE.
  Start ComfyUI:   cd $W/ComfyUI && python main.py --listen 0.0.0.0 --port 8188
                   (open port 8188 via Vast "Open" button / SSH tunnel: ssh -L 8188:localhost:8188 ...)
                   Load the Wan 2.2 I2V / S2V templates from ComfyUI's Workflow > Browse Templates.
  Audio:           cd $W/audio_pipeline && python pipeline.py your_project.json
  Before you leave: copy outputs off the box, then STOP or DESTROY the instance.
EOF
