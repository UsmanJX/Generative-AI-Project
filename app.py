# app.py
# Streamlit interface for Roman‑coin reconstruction with three GAN families
# --------------------------------------------------
import streamlit as st
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# --------------------------------------------------
# 1) Model registry
# --------------------------------------------------
REPO_ROOT = Path(__file__).parent
MODEL_DIR = REPO_ROOT / "models"
SRC_DIR   = REPO_ROOT / "src"

import sys
sys.path.append(str(SRC_DIR))   # so we can import local modules

# from congan import Generator             # noqa: E402
from cyclegan import Generator  # noqa: E402
from stylegan import StyleGANGenerator         # noqa: E402

@st.cache_resource(show_spinner=False)
def load_generators(device="cpu"):
    """Load all GAN generators once and keep them in memory."""
    gens = {}

    # ConGAN ------------------------------------------------------------
    gen_c = Generator().to(device).eval()
    gen_c.load_state_dict(torch.load(MODEL_DIR / "ConGan.pth",
                                     map_location=device))
    gens["ConGAN (128 px)"] = {"net": gen_c, "z_dim": 100, "res": 128}

    # Encoder–Cycle GAN --------------------------------------------------
    gen_e = Generator().to(device).eval()
    gen_e.load_state_dict(torch.load(MODEL_DIR / "CycleGan.pth",
                                     map_location=device))
    gens["Encoder‑Cycle GAN (128 px)"] = {"net": gen_e, "z_dim": 100, "res": 128}

    # StyleGAN -----------------------------------------------------------
    gen_s = Generator().to(device).eval()
    gen_s.load_state_dict(torch.load(MODEL_DIR / "StyleGan.pth",
                                     map_location=device))
    gens["StyleGAN (256 px)"] = {"net": gen_s, "z_dim": 512, "res": 256}

    return gens

# --------------------------------------------------
# 2) Pre‑processing helpers (same as notebooks)
# --------------------------------------------------
def segment_and_crop(img_bgr: np.ndarray, target: int) -> Image.Image:
    """Replicates the OpenCV‑based circular crop used in training."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contour detected – try another image.")

    # largest contour is assumed to be the coin rim
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    crop = img_bgr[y:y + h, x:x + w]

    # make square (pad shorter side)
    diff = abs(h - w)
    if h > w:
        pad = ((0, 0), (diff // 2, diff - diff // 2), (0, 0))
    else:
        pad = ((diff // 2, diff - diff // 2), (0, 0), (0, 0))
    crop_sq = np.pad(crop, pad, mode="constant", constant_values=0)

    crop_rgb = cv2.cvtColor(crop_sq, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(crop_rgb).resize((target, target),
                                                Image.BILINEAR)
    return pil_img

# Normalisation used during training ([-1,1] space)
norm = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
])

denorm = T.Compose([
    T.Normalize(mean=[-1, -1, -1],
                std=[2, 2, 2]),
    T.ToPILImage()
])

# --------------------------------------------------
# 3) Streamlit page layout
# --------------------------------------------------
st.set_page_config(page_title="Roman‑Coin Reconstruction",
                   layout="centered",
                   page_icon="🪙")

st.title("🪙 Roman‑Coin Reconstruction with GANs")
st.markdown(
    "Upload a photograph of an ancient coin, choose the reconstruction "
    "model, and the app will generate a cleaned / hallucinated version "
    "based on the GAN you select. Everything runs locally – no images "
    "leave your machine."
)

device = "cuda" if torch.cuda.is_available() else "cpu"
generators = load_generators(device=device)

model_name = st.sidebar.selectbox(
    "Choose model",
    list(generators.keys()),
    index=2  # default StyleGAN
)

uploaded = st.file_uploader("Drop a coin image (JPEG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # --------------------------------------------------
    # 4) Pre‑process
    # --------------------------------------------------
    raw = Image.open(uploaded).convert("RGB")
    gen_info = generators[model_name]
    target   = gen_info["res"]

    try:
        preproc_pil = segment_and_crop(np.array(raw)[:, :, ::-1], target)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    # --------------------------------------------------
    # 5) Run inference
    # --------------------------------------------------
    gen = gen_info["net"]
    z_dim = gen_info["z_dim"]

    with torch.no_grad():
        z = torch.randn(1, z_dim, device=device)
        if "StyleGAN" in model_name:
            fake = gen(z)                          # StyleGAN signature
        else:
            label = torch.zeros(1, dtype=torch.long, device=device)  # dummy class 0
            fake = gen(z, label)                   # Con/Encoder signature

    fake_img = denorm(fake.squeeze().cpu().clamp_(-1, 1)).resize(raw.size)

    # --------------------------------------------------
    # 6) Display
    # --------------------------------------------------
    col1, col2 = st.columns(2)
    col1.header("Original (cropped)")
    col1.image(preproc_pil, use_column_width=True)

    col2.header("Reconstruction")
    col2.image(fake_img, use_column_width=True)

    st.success("Done! Feel free to switch models for the same upload.")

# --------------------------------------------------
# 7) Footer
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Built with PyTorch and Streamlit.  © 2025")

