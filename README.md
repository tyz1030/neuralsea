# Beyond NeRF Underwater: Learning Neural Reflectance Fields for True Color Correction of Marine Imagery
coming soon\
paper in submission\
[arxiv](https://arxiv.org/abs/2304.03384)

Left: with water effects;       Right: color corrected \
![novel view](https://github.com/tyz1030/neuralsea/blob/38b7be23c4f21d43948723782a2576945ddd40ee/novelview.gif "novel view")\
Groundtruth image\
![gt](https://github.com/tyz1030/neuralsea/blob/67a378e6be1095d925fa044a1a5e7d6566f11340/raw081.png "gt")


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
## Data ##
We have [white lego dataset](https://drive.google.com/drive/folders/1wy5nqjScpv-IhXK34UyBTfYI8LFjChcZ?usp=share_link "white lego") released.

(we will release rest of the dataset and weights once the manuscript is finalized)

Download 'data_lego_white' and put it under the neuralsea directory.

Download [white lego weights](https://drive.google.com/file/d/1t8dh7cV-m5r86lLvkS7ft8tSdnFaQMfx/view?usp=sharing) and put it under neuralsea/checkpoints/ to use our pretrained weights.

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
python3 train_nerf.py --config-name legow
python3 test_nerf.py --config-name legow
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
[arxiv](https://arxiv.org/abs/2304.03384)
```
@misc{zhang2023nerf,
      title={Beyond NeRF Underwater: Learning Neural Reflectance Fields for True Color Correction of Marine Imagery}, 
      author={Tianyi Zhang and Matthew Johnson-Roberson},
      year={2023},
      eprint={2304.03384},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
