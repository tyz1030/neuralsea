# NeuralSea
coming soon\
paper in submission

## Data ##
We have [white lego dataset](https://drive.google.com/drive/folders/1wy5nqjScpv-IhXK34UyBTfYI8LFjChcZ?usp=share_link "white lego") released.

Download 'data_lego_white' and put it under the neuralsea directory.

Download [white lego weights](https://drive.google.com/file/d/1t8dh7cV-m5r86lLvkS7ft8tSdnFaQMfx/view?usp=sharing) and put it under neuralsea/checkpoints/ to use our pretrained weights.

## Train & Test ##
```
python3 train_nerf.py --config-name legow
python3 test_nerf.py --config-name legow
```
