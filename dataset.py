# %%
from decorators import timing
from utils import merge_dicts
from email.mime import audio
from fileinput import filename
from heapq import merge
import json
import sys
import os
import shutil
from joblib import Parallel, delayed

import librosa
import numpy as np
import pandas as pd
import scipy.io as sio
from praudio import utils
from sklearn.model_selection import train_test_split
from plot import plot_class_distribution
import eyed3

eyed3.log.setLevel("ERROR")


DATASET_PATH = "../datasets/base_portuguese"
OUTPUTDIR_PATH = "./dataset/base"
SAMPLES_TO_CONSIDER = 22050  # 1 sec. of audio
SEED = 42


@timing
def annotate_dataset(
    dataset_path: str,
    output_path: str,
    sr: int = 24000,
    extract: int = None,
    plot_distribution=False,
):
    """
    It takes in a dataset path and outputs a metadata.csv file and a folder of audio files.

    :param dataset_path: The path to the dataset folder
    :type dataset_path: str
    :param output_path: the path where the audio files will be saved
    :type output_path: str
    :param sample_rate: The sample rate of the audio files, defaults to 24000
    :type sample_rate: int (optional)
    """

    base_data = {
        "label": [],
        "sample_rate": [],
        "length": [],
        "artists": [],
        "album": [],
        "title": [],
        "genre": [],
        "filename": [],
    }

    utils.create_dir_hierarchy(output_path + "/audio")

    # loop through all sub-dirs
    def _run(i, filepath):
        # for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        data = {
            "label": [],
            "sample_rate": [],
            "length": [],
            "artists": [],
            "album": [],
            "title": [],
            "genre": [],
            "filename": [],
        }

        try:
            dirpath = filepath.split("/")[:-1]
            filename = filepath.split("/")[-1]

            filepath_wav = filepath.replace("mp3", "wav")
            filename_wav = filename.replace(".mp3", ".wav")

            if not os.path.exists(filepath_wav):
                raise FileNotFoundError(f"{i} - {filepath_wav} does not exist")

            signal, sample_rate = librosa.load(filepath_wav, sr=sr, mono=True)

            audioinfo = eyed3.load(filepath)

            if audioinfo is None:
                return data

            sio.wavfile.write(
                f"{output_path}/audio/{filename_wav}", sample_rate, signal
            )

            if not os.path.exists(f"{output_path}/audio/{filename_wav}"):
                raise Exception(f"Files not copied or created {filename}")

            data["artists"].append(audioinfo.tag.artist)
            data["album"].append(audioinfo.tag.album)
            data["title"].append(audioinfo.tag.title)
            data["genre"].append(
                audioinfo.tag.genre.name if audioinfo.tag.genre else "unknown"
            )

            data["sample_rate"].append(sample_rate)
            data["length"].append(int((len(signal) / sample_rate) * 1000))
            data["label"].append(i)
            data["filename"].append(filename_wav)

            return data
        except RuntimeError:
            print(f"{filename} does not have an image")
        # except FileNotFoundError as e:
        #     print(f'{filename} - {e}')
        #     print(f'Removing {filepath}...')
        #     # os.remove(filepath)
        #     # shutil.rmtree(f'{output_path}', ignore_errors=True)
        #     return sys.exit(1)

    filename_wav = []
    filename_mp3 = []

    for dirpath, _, filenames in os.walk("/src/spotify/wav"):
        for f in filenames:
            if f.endswith(".wav"):
                filename_wav.append(f"{os.path.splitext(f)[0]}")

    for dirpath, _, filenames in os.walk("/src/spotify/mp3"):
        for f in filenames:
            if f.endswith(".mp3"):
                filename_mp3.append(f"{os.path.splitext(f)[0]}")

    filename_list = list(set(filename_wav).intersection(set(filename_mp3)))

    # dicts = [_run(i, filename) for i, filename in enumerate(
    #     filename_list) if extract and i < extract]

    dicts = Parallel(n_jobs=-1)(
        delayed(_run)(i, f"{dataset_path}/{filename}.mp3")
        for i, filename in enumerate(filename_list)
        if extract and i < extract
    )

    data = merge_dicts(base_data, *dicts)

    # if plot_class_distribution:
    #     labels, count = np.unique(data['label'], return_counts=True)

    #     plot_class_distribution(labels,
    #                             count, save_path=f'{output_path}')

    pd.DataFrame.from_dict(data).to_csv(f"{output_path}/metadata.csv", index=False)


# %%
if __name__ == "__main__":
    n_classes = 20

    shutil.rmtree(f"/src/tcc_netro/dataset/spotify_{n_classes}", ignore_errors=True)

    annotate_dataset(
        "/src/spotify/mp3",
        f"/src/tcc_netro/dataset/spotify_{n_classes}",
        extract=n_classes,
    )

    # annotate_inference('/src/datasets/ifgaudio',
    #                    '/src/tcc/dataset/inference',
    #                    '/src/tcc/models/base_portuguese_40/SEG_1/MFCC_18/1649037587_71.875',
    #                    '/src/tcc/catalog.csv')
    # prepare_raw_dataset(DATASET_PATH, OUTPUTDIR_PATH)
    # split_dataset('/src/tcc_devbaraus/dataset/base', '/src/tcc_devbaraus/dataset', True)

# %%
