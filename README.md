# HAM-Net

Code for HAM-Net: **A Hybrid Attention Mechanism for Weakly-Supervised Temporal Action Localization**


## Prerequisites

Pytorch-1.5+, pytorch_lightning-1.1.*, loguru, colorama, etc.

You can create a new conda environment using:
```
conda env create -f environment.yml
```

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
