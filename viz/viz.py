
import math
from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

import torch
from torch import optim, nn
from torch.nn import functional as F
import torchaudio
import torchaudio.transforms as T
import librosa 
from einops import rearrange

import wandb
import numpy as np
import pandas as pd

def spectrogram_image(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None, db_range=[35,120]):
    """
    # cf. https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html

    """
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axs = fig.add_subplot()
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect, vmin=db_range[0], vmax=db_range[1])
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba)


def audio_spectrogram_image(waveform, power=2.0, sample_rate=48000):
    """
    # cf. https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html
    """
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 80

    mel_spectrogram_op = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, 
        hop_length=hop_length, center=True, pad_mode="reflect", power=power, 
        norm='slaney', onesided=True, n_mels=n_mels, mel_scale="htk")

    melspec = mel_spectrogram_op(waveform.float())
    melspec = melspec[0] # TODO: only left channel for now
    return spectrogram_image(melspec, title="MelSpectrogram", ylabel='mel bins (log freq)')
