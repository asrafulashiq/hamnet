<div align="center">    

# HAM-Net

<p align="center">
  <img src="data/hamnet_model.png" width="600">
</p>


</div>


Code for HAM-Net: **A Hybrid Attention Mechanism for Weakly-Supervised Temporal Action Localization**

[Paper](https://drive.google.com/file/d/16z4PyE_t6n6O1akN2OeAOVYemif7A8Nd/view?usp=sharing)

## Prerequisites

PyTorch-1.7.1, pytorch_lightning-1.1.2, loguru, colorama, etc. Older versions of PyTorch(1.3+) and pytorch-lightning(0.9+) should also work but not tested. 

You can create a new conda environment with all the dependencies using:
```
conda env create -f environment.yml
```

## How to Run

### Download Data

The ground-truth and I3D features for THUMOS14 and ActivitiNet1.2 dataset can be downloaded from here:

[Box Download Link](https://rpi.box.com/s/hf6djlgs7vnl7a2oamjt0vkrig42pwho)

Please put the downloaded files/folders under `data/` directory.

### Training

To train HAM-Net on *Thumos14* dataset:

```python

python main.py
```

### Testing

To evaluate on *Thumos14* dataset:

```python

python main.py --test --ckpt [checkpoint_path]
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