# Transformer with Pytorch

* Pytorch Impementation [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

## DataAugmentation

|   |  negative |  negative |  negative | negative  |
|---|---|---|---|---|
|  positive |  ![6](source/positive/6.png) |  ![7](source/positive/7.png) |  ![8](source/positive/8.png) |  ![9](source/positive/9.png) |
|  positive |  ![1](source/positive/1.png) |  ![2](source/positive/2.png) |  ![3](source/positive/3.png) |  ![4](source/positive/4.png) |

## Dependency

```
python-opencv
torch
torchvision
matplotlib
pandas
PIL
```

## Directory
```
├── data
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── ...
│   ├── 5.jpg
│   └── 6.jpg
├── .gitignore
├── dataset_wrapper.py
├── model.py
├── nt_xent_loss.py
├── README.md
└── simclr.py
```

## RUN

```
(pytorch) xxx:xxx xxx$ python simclr.py 
----------------
epoch : 0
loss  : 1.5892890691757202
lr    : 0.0003
----------------
epoch : 1
loss  : 1.5985093116760254
lr    : 0.00029699999999999996
----------------
...
----------------
epoch : 5
loss  : 0.6665001213550568
lr    : 0.00028529701496999996
----------------
epoch : 6
loss  : 0.5047109872102737
lr    : 0.0002824440448203
----------------
epoch : 7
loss  : 0.4476690739393234
lr    : 0.00027961960437209696
----------------
```

## Reference 

* [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
* [https://github.com/sthalles/SimCLR](https://github.com/sthalles/SimCLR)