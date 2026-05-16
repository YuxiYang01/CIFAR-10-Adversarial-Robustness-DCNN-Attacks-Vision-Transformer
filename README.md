# CIFAR-10 Adversarial Robustness
Deep learning project on CIFAR-10 covering adversarial attacks, defense, and a Vision Transformer comparison. A custom ResNet-style DCNN achieves 95.15% test accuracy. Three attack methods (Gaussian Noise, FGSM, PGD) are evaluated at ε ∈ {0.01, 0.03, 0.10}, with PGD-based adversarial training as defense. Includes a from-scratch Tiny ViT implementation. Built with PyTorch on Google Colab.

## Part A — Deep Convolutional Neural Network

### Architecture

Custom ResNet-style CNN with four residual stages:

| Component | Channels | Stride | Output |
|-----------|----------|--------|--------|
| Stem (Conv3×3 + BN + ReLU) | 64 | 1 | 32×32 |
| Stage 1 (2× ResBlock) | 64 | 1 | 32×32 |
| Stage 2 (2× ResBlock) | 128 | 2 | 16×16 |
| Stage 3 (2× ResBlock) | 256 | 2 | 8×8 |
| Stage 4 (2× ResBlock) | 512 | 2 | 4×4 |
| GAP + Dropout(0.4) + FC | 10 | — | 10 |

**Parameters:** ~11.17M

### Training Configuration

- **Optimizer:** SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
- **Scheduler:** Cosine annealing over 100 epochs
- **Loss:** Cross-entropy with label smoothing (0.1)
- **Augmentation:** RandomCrop(32, pad=4), RandomHorizontalFlip
- **GPU:** Tesla T4 (Google Colab)

### Result

| Metric | Value |
|--------|-------|
| Train accuracy | 99.98% |
| **Test accuracy** | **95.15%** |


## Part B — Adversarial Attacks & Defense

### Attack Methods

**Gaussian Noise** — Baseline non-adversarial perturbation. Adds random noise δ ~ N(0, ε²) in pixel space.

**FGSM** (Goodfellow et al., 2014) — Single-step gradient attack:

$$x_{adv} = x + \varepsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y))$$

**PGD** (Madry et al., 2018) — Iterative attack with 20 steps, step size α = ε/4, random start:

$$x^{(t+1)} = \Pi_{B(x, \varepsilon)} \left[ x^{(t)} + \alpha \cdot \text{sign}(\nabla_x \mathcal{L}) \right]$$

### Attack Results

Clean accuracy baseline: **95.15%**

| Attack | ε = 0.01 | ε = 0.03 | ε = 0.10 |
|--------|----------|----------|----------|
| Noise | 94.69% | 84.69% | 23.05% |
| FGSM | 74.45% | 63.91% | 15.08% |
| PGD | 58.83% | 31.41% | **2.58%** |


### Defense: Adversarial Training

PGD-based adversarial training (Madry et al., 2018). Fine-tuned from Part A checkpoint for 30 epochs on a 50/50 mix of clean and PGD adversarial examples (ε=0.03, 7 steps). Best robust clean accuracy: **88.58%**.

| Attack | ε | Standard | Robust | Δ |
|--------|---|----------|--------|---|
| Noise | 0.01 | 94.69% | 88.83% | −5.86 |
| Noise | 0.03 | 84.69% | 88.36% | +3.67 |
| Noise | 0.10 | 23.05% | 82.73% | **+59.68** |
| FGSM | 0.01 | 74.45% | 78.67% | +4.22 |
| FGSM | 0.03 | 63.91% | 55.70% | −8.21 |
| FGSM | 0.10 | 15.08% | 14.84% | −0.24 |
| PGD | 0.01 | 58.83% | 77.81% | **+18.98** |
| PGD | 0.03 | 31.41% | 44.53% | +13.12 |
| PGD | 0.10 | 2.58% | 0.55% | −2.03 |

## Part C (Extra Credit) — Vision Transformer

### Architecture

From-scratch Tiny ViT designed for CIFAR-10:

| Parameter | Value |
|-----------|-------|
| Input size | 96×96 (resized from 32×32) |
| Patch size | 8×8 (144 patches) |
| Embedding dim | 192 |
| Transformer depth | 9 blocks |
| Attention heads | 3 |
| MLP ratio | 4.0 |
| Parameters | **4.07M** |

**Training:** AdamW (lr=1e-3, weight_decay=0.05), 10-epoch warmup + cosine decay, RandAugment, 100 epochs.

### DCNN vs ViT

| Model | Parameters | Test Accuracy |
|-------|------------|---------------|
| DCNN (ResNet-style) | ~11.17M | **95.15%** |
| ViT (Tiny, from scratch) | 4.07M | **88.21%** |

The ~7pp gap is expected — ViTs lack CNN inductive biases (locality, translation equivariance) and require substantially more data to compensate (Dosovitskiy et al., 2021).

---

## Quick Start

1. Open `CIFAR10_DCNN_Adversarial.ipynb` in Google Colab
2. Set runtime to **GPU (T4)**
3. Run Cell 0 to mount Google Drive (checkpoints auto-save)
4. Run cells sequentially — CIFAR-10 downloads automatically

### Estimated Runtime (T4 GPU)

| Part | Time |
|------|------|
| Part A (100 epochs) | ~25 min |
| Part B attacks | ~10 min |
| Part B defense (30 epochs) | ~45 min |
| Part C ViT (100 epochs) | ~40 min |

## Requirements

```
torch >= 2.0
torchvision
timm
numpy
matplotlib
```

## References

- Goodfellow, I. et al. (2014). *Explaining and Harnessing Adversarial Examples.* arXiv:1412.6572
- Madry, A. et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018
- Dosovitskiy, A. et al. (2021). *An Image is Worth 16x16 Words.* ICLR 2021

## License

MIT
