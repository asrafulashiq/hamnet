<div align="center">    

# HAM-Net

<p align="center">
  <img src="data/hamnet_model.png" width="400">
</p>


</div>


Code for HAM-Net: **A Hybrid Attention Mechanism for Weakly-Supervised Temporal Action Localization**


## Prerequisites
---
**Pytorch-1.5+**, **pytorch_lightning-1.1.***, loguru, colorama, etc.

You can create a new conda environment with all dependencies using:
```
conda env create -f environment.yml
```

## How to Run
---
### Training

To run HAM-Net on *Thumos14* dataset:

```python

python main.py
```

### Testing

To evaluate on *Thumos14* dataset:

```python

python main.py --ckpt [checkpoint_path]
```

For ActivityNet-1.2, use `main_anet.py` script.


## Citation
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
``` 