import numpy as np

from decorators import timing


def merge_dicts(base_dict, *dicts):
    """
    Merge two dictionaries.
    """
    dict3 = base_dict.copy()

    for d in dicts:
        for key, value in d.items():
            dict3[key] = [*dict3[key], *value]

    return dict3


@timing
def convert_wav_to_opus(folder):
    """
    Convert all files in a folder to Opus format.
    """
    import os
    import subprocess

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            subprocess.call(
                [
                    "ffmpeg",
                    "-i",
                    f"{folder}/{file}",
                    f"{folder}/{os.path.splitext(file)[0]}.opus",
                ]
            )
            os.remove(f"{folder}/{file}")


def arr_dimen(a):
    return [len(a)] + arr_dimen(a[0]) if (type(a) == list) else []


if __name__ == "__main__":
    base = {
        "a": [],
        "b": [],
        "c": [],
    }

    dic1 = {
        "a": [1, 2, 3],
        "b": [1, 2, 3],
        "c": [1, 2, 3],
    }

    dic2 = {
        "a": [4, 5, 6],
        "b": [4, 5, 6],
        "c": [4, 5, 6],
    }

    final_dict = merge_dicts(base, *[dic1, dic2])

    print(final_dict)
