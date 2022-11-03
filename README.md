# Emotional VITS

[//]: # ([![Hugging Face Spaces]&#40;https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue&#41;]&#40;https://huggingface.co/spaces/innnky/nene-emotion&#41; )

[bilibili demo](https://www.bilibili.com/video/BV1Me4y1m7aT/)

数据集无需任何情感标注，通过[情感提取模型](https://github.com/audeering/w2v2-how-to) 提取语句情感logits输入网络，实现情感可控的VITS合成
## 模型结构
+ 将情感embedding（1024维）替换为logits（3维）输入模型，由于logits仅包含arousal, dominance, valence 三个值，因此可以手动调整值进行合成，而不需要使用参考音频提取emb做情感参考

## 相较于输入embedding的优缺点介绍
使用logits控制情感的优点：
+ 推理更加易用，只需要手动设置三个参数即可，不需要参考音频，例如arousal参数控制情感强度，取值0-1，1则为最强烈的情感，0为最轻柔的情感
+ 由于不需要指定参考音频，因此做多说话人的模型会方便很多，可以有统一的参数控制不同说话人的情感

使用logits控制情感的缺点：
+ 只有3个参数可以控制，无法精确的还原所需要的情感
+ 除了arousal参数控制情感的强度效果明显，另外两个参数感觉控制性比较弱，修改参数后合成出来效果没啥区别

## 注意事项
+ 模型预处理（提取情感）和训练时都需要指定--emotion-type logits
## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
0. prepare datasets
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for nene have been already provided.
python preprocess.py --text_index 2 --filelists filelists/train.txt filelists/val.txt --text_cleaners japanese_cleaners


```
5. extract emotional embeddings, this will generate *.logits.npy for each wav file.
```sh
python emotion_extract.py --filelists filelists/train.txt filelists/val.txt --emotion-type logits
```


## Training Exmaple
```sh

# nene
python train_ms.py -c configs/nene.json -m nene  --emotion-type logits

# if you are fine tuning pretrained original VITS checkpoint ,
python train_ms.py -c configs/nene.json -m nene --emotion-type logits --ckptD /path/to/D_xxxx.pth --ckptG /path/to/G_xxxx.pth

```


## Inference Example
See [inference-logits.ipynb](inference-logits.ipynb) 
