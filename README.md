# HAM-Net

Code for HAM-Net: **A Hybrid Attention Mechanism for Weakly-Supervised Temporal Action Localization**


## Prerequisites

Pytorch-1.3+, pytorch_lightning-0.9.0, loguru, colorama, etc.

## Training

To run HAM-Net on *Thumos14* dataset:

```python

python main.py
```

## Testing

To evaluate on *Thumos14* dataset:

```python

python main.py --ckpt [checkpoint_path]
```

For ActivityNet-1.2, use `main_anet.py` script.
