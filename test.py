# %%
import tensorflow as tf
import librosa.display
import tensorflow_io as tfio
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.io as sio
import os

from utils import convert_files_to_opus


# %%
audio, sample_rate = librosa.load("/src/spotify/opus/18 Dollars - flora cash.opus")
#%%
sf.write("./base.ogg", audio, sample_rate)

# %%
sio.wavfile.write

# %%
convert_files_to_opus("/src/tcc_netro/dataset/spotify_10/audio")

# %%
