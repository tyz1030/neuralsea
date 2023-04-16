# NeuralSea
coming soon\
paper ([arxiv](https://arxiv.org/abs/2304.03384)) in submission

Left: with water effects;       Right: color corrected \
![novel view](https://github.com/tyz1030/neuralsea/blob/38b7be23c4f21d43948723782a2576945ddd40ee/novelview.gif "novel view")\
Groundtruth image\
![gt](https://github.com/tyz1030/neuralsea/blob/67a378e6be1095d925fa044a1a5e7d6566f11340/raw081.png "gt")

## Data ##
We have [white lego dataset](https://drive.google.com/drive/folders/1wy5nqjScpv-IhXK34UyBTfYI8LFjChcZ?usp=share_link "white lego") released.

(we are working on releasing rest of the dataset and weights)

Download 'data_lego_white' and put it under the neuralsea directory.

Download [white lego weights](https://drive.google.com/file/d/1t8dh7cV-m5r86lLvkS7ft8tSdnFaQMfx/view?usp=sharing) and put it under neuralsea/checkpoints/ to use our pretrained weights.

## Train & Test ##
```
python3 train_nerf.py --config-name legow
python3 test_nerf.py --config-name legow
```

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
