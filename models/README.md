# 🎤 SonicShift - AI Voice Models Directory

This directory stores neural weights for voice conversion.

---

## Directory Structure

```text
models/
├── pretrained/         # Base feature & pitch extraction models
│   ├── hubert_base.pt  # ContentVec speech encoder (~180 MB)
│   └── rmvpe.pt        # Neural pitch estimator (~38 MB)
└── checkpoints/        # User-trained or downloaded voice models
    ├── your_model.pth  # RVC generator weights
    └── your_model.index# (Optional) FAISS feature index
```

---

## 1. Download Base Pretrained Weights
To automatically fetch `hubert_base.pt` and `rmvpe.pt`:

```powershell
python scripts/download_models.py
```

---

## 2. Where to Find Free Female Voice Models
You can download RVC v2 model checkpoints from:
- **Weights.gg** ([weights.gg](https://weights.gg/)) - Search for anime, Vtuber, or streamer voices.
- **Hugging Face** ([huggingface.co/models?search=rvc](https://huggingface.co/models?search=rvc)) - Open-source RVC repositories.
- **AI Hub Discord** - Free community models.

---

## 3. How to Use Your Downloaded Model
1. Copy the `.pth` file and `.index` file into `models/checkpoints/`.
2. Run SonicShift specifying the model path:

```powershell
python main.py --model "models/checkpoints/your_model.pth" --index "models/checkpoints/your_model.index"
```
