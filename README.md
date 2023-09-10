# Beyond NeRF Underwater: Learning Neural Reflectance Fields for True Color Correction of Marine Imagery
[**UPDATE!**] Our paper just gets accepted by RA-L! Thanks to reviewers, collaborators, editors, and those who raise issues on GitHub which helped us improve the code a lot.

All [data](https://drive.google.com/drive/folders/1JU9AXmWA-18B1rg6Aj2gYXD2_nE06B7h?usp=drive_link) presented in the paper are now released.\
Example: Download 'data_lego_white' and put it under the neuralsea directory. Download [white lego weights](https://drive.google.com/file/d/1M8zOzWOWT06GyotxbYN1Zba7vqMJCOU6/view?usp=drive_link) and put it under neuralsea/checkpoints/ to use our pretrained weights (not all weights are released so far. I will upload them once I got time. Feel free to train from scratch and I don't usually stick with certain random seeds.).


## Publication ##
Our paper is published at [IEEE RA-L](https://ieeexplore.ieee.org/document/10225666). You can also find it on [arxiv](https://arxiv.org/abs/2304.03384). To appear on ICRA 2024.\
This work is supported by National Oceanic and Atmospheric Administration (NOAA) under grant NA22OAR0110624.

## Visualizations ##
Left: with water effects;       Right: color corrected \
![novel view](https://github.com/tyz1030/neuralsea/blob/38b7be23c4f21d43948723782a2576945ddd40ee/novelview.gif "novel view")\
Groundtruth image\
![gt](https://github.com/tyz1030/neuralsea/blob/67a378e6be1095d925fa044a1a5e7d6566f11340/raw081.png "gt")

![novel view](https://github.com/tyz1030/neuralsea/blob/79eb82be6a3263f10813ba84d088d468ff5a3b48/yellow_lego.gif "novel view")\
Groundtruth image\
![gt](https://github.com/tyz1030/neuralsea/blob/8bc04ff98ac2265d1688bb840d96f3ddbb9d286a/raw081_yellow.png "gt")


More real-world results:\
Lake Erie:\
![novel view](https://github.com/tyz1030/neuralsea/blob/254659d2bf3547b158b140788775030911940a3c/erie01.gif "novel view")\
![novel view](https://github.com/tyz1030/neuralsea/blob/254659d2bf3547b158b140788775030911940a3c/erie02.gif "novel view")\
Water Tank Low Turbidity:\
![novel view](https://github.com/tyz1030/neuralsea/blob/83d8da0f5273690b592ff8f6d95c346937f483c1/lab41_lowturbidity.gif "novel view")\
Water Tank Mid Turbidity:\
![novel view](https://github.com/tyz1030/neuralsea/blob/83d8da0f5273690b592ff8f6d95c346937f483c1/lab42_midturbidity.gif "novel view")\
Water Tank High Turbidity:\
![novel view](https://github.com/tyz1030/neuralsea/blob/83d8da0f5273690b592ff8f6d95c346937f483c1/lab43_highturbidity.gif "novel view")

<!---
## Data ##
We have [white lego dataset](https://drive.google.com/drive/folders/1wy5nqjScpv-IhXK34UyBTfYI8LFjChcZ?usp=share_link "white lego") released.

(we will release rest of the dataset and weights once the manuscript is finalized)

Download 'data_lego_white' and put it under the neuralsea directory.

Download [white lego weights](https://drive.google.com/file/d/1t8dh7cV-m5r86lLvkS7ft8tSdnFaQMfx/view?usp=sharing) and put it under neuralsea/checkpoints/ to use our pretrained weights.
-->

## Dependencies ##

Install [PyTorch](https://pytorch.org/get-started/locally/):
```
pip install torch torchvision
```

Install PyTorch3D, please follow their [instruction](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). We use the following to install:
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
Install other dependencies:
```
pip install hydra-core plotly visdom matplotlib
```

## Train & Test ##
```
# for synthetic data example
python3 train_nerf.py --config-name synthetic_lego_white
python3 test_nerf.py --config-name synthetic_lego_white
```

```
# for water tank data example
python3 train_nerf.py --config-name real_watertank
python3 test_nerf.py --config-name real_watertank
```

## Visualization ##
install visdom
```
pip install visdom
```
run visdom
```
visdom
```
Then in your browser, navigate to http://localhost:8097/


## Paper ##
If you find this study helpful please kindly cite us:
```
@ARTICLE{10225666,
  author={Zhang, Tianyi and Johnson-Roberson, Matthew},
  journal={IEEE Robotics and Automation Letters}, 
  title={Beyond NeRF Underwater: Learning Neural Reflectance Fields for True Color Correction of Marine Imagery}, 
  year={2023},
  volume={8},
  number={10},
  pages={6467-6474},
  doi={10.1109/LRA.2023.3307287}}
```
