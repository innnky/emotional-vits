import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import os
import librosa
import numpy as np


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


# load model from hub
device = 'cuda' if torch.cuda.is_available() else "cpu"
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)


def process_func(
        x: np.ndarray,
        sampling_rate: int,
        embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y


#
#
# def disp(rootpath, wavname):
#     wav, sr = librosa.load(f"{rootpath}/{wavname}", 16000)
#     display(ipd.Audio(wav, rate=sr))

rootpath = "dataset/nene"
embs = []
wavnames = []


def extract_dir(path):
    rootpath = path
    for idx, wavname in enumerate(os.listdir(rootpath)):
        wav, sr = librosa.load(f"{rootpath}/{wavname}", 16000)
        emb = process_func(np.expand_dims(wav, 0), sr, embeddings=True)
        embs.append(emb)
        wavnames.append(wavname)
        np.save(f"{rootpath}/{wavname}.emo.npy", emb.squeeze(0))
        print(idx, wavname)


def extract_wav(path):
    wav, sr = librosa.load(path, 16000)
    emb = process_func(np.expand_dims(wav, 0), sr, embeddings=True)
    return emb


def preprocess_one(path):
    wav, sr = librosa.load(path, 16000)
    emb = process_func(np.expand_dims(wav, 0), sr, embeddings=True)
    np.save(f"{path}.emo.npy", emb.squeeze(0))
    return emb


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Emotion Extraction Preprocess')
    parser.add_argument('--filelists', dest='filelists',nargs="+", type=str, help='path of the filelists')
    args = parser.parse_args()

    for filelist in args.filelists:
        print(filelist,"----start emotion extract-------")
        with open(filelist) as f:
            for idx, line in enumerate(f.readlines()):
                path = line.strip().split("|")[0]
                preprocess_one(path)
                print(idx, path)
