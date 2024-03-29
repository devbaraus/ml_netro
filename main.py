# %%
import argparse
import json
import os
import pickle as pkl
import time

import audiomentations as am
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
from praudio import utils
from sklearn.preprocessing import StandardScaler

from dataset import annotate_dataset
from loaders import load_mat_representation
from plot import plot_class_distribution, plot_confusion_matrix, plot_history
from preprocess import represent_dataset, segment_dataset
from train import build_perceptron, train_model

gpu_devices = tf.config.experimental.list_physical_devices("GPU")

for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

parser = argparse.ArgumentParser(description="Arguments algo")

parser.add_argument(
    "-c",
    type=int,
    action="store",
    dest="coeff",
    required=False,
    help="Coeficientes",
    default=None,
)

parser.add_argument(
    "-a",
    type=int,
    action="store",
    dest="augmentation",
    required=False,
    help="Augmentation",
    default=None,
)

parser.add_argument(
    "-s",
    type=int,
    action="store",
    dest="segment",
    required=False,
    help="Segment ime",
    default=None,
)

parser.add_argument(
    "-o",
    type=float,
    action="store",
    dest="overlap",
    required=False,
    help="Overalp data",
    default=None,
)

parser.add_argument(
    "-b",
    type=str,
    action="store",
    dest="base",
    required=False,
    help="Dataset",
    default=None,
)


args, _ = parser.parse_known_args()

print(args)

# %%
BASE_DATASETS = "/src/"
ANNOTATE_DIR = "/src/tcc_netro/dataset"
MODELS_DIR = "/src/tcc_netro/models"
DATASET_DIR = args.base or "spotify_20"

ANNOTATE_DATASET = False
SEGMENT_TEST = True
SEGMENT_TRAIN = True
SEGMENT_VALID = True
REPRESENT_TEST = True
REPRESENT_TRAIN = True
REPRESENT_VALID = True

# MODEL ARCHITECTURE
MODEL_DENSE_1 = 320
MODEL_DROPOUT_1 = 0.3
MODEL_DENSE_2 = 310
MODEL_DROPOUT_2 = 0
MODEL_DENSE_3 = 0


# MODEL TRAINING
EPOCHS = 1000
BATCH_SIZE = 32
PATIENCE = 20
LEARNING_RATE = 0.0001

# SEGMENTATION
SAMPLE_RATE = 24000
SEGMENT_LENGTH = args.segment or 5
OVERLAP_SIZE = args.overlap or 0.0
AUGMENT_SIZE = args.augmentation or 0

# REPRESENTATION
MFCC_COEFF = args.coeff or 40
MFCC_N_FFT = 2048
MFCC_HOP_LENGTH = 512

# %%
if ANNOTATE_DATASET and not os.path.exists(f"{ANNOTATE_DIR}/{DATASET_DIR}"):
    annotate_dataset(
        f"{BASE_DATASETS}/{DATASET_DIR}",
        f"{ANNOTATE_DIR}/{DATASET_DIR}",
        SAMPLE_RATE,
        plot_distribution=True,
    )

# %%
BASE_TRANSFORM = [
    am.Normalize(p=1),
]

TRAIN_TRANSFORM = [
    am.AddBackgroundNoise(
        "/src/datasets/ESC-50 Dataset for Environmental Sound Classification/ESC-50-master/resampled",
        p=0.5,
    ),
    am.AddGaussianSNR(min_snr_in_db=24, max_snr_in_db=40, p=0.8),
    am.HighPassFilter(min_cutoff_freq=60, max_cutoff_freq=100, p=0.8),
    am.LowPassFilter(min_cutoff_freq=3400, max_cutoff_freq=10000, p=0.8),
]

# %%
BASE_DIR = f"{ANNOTATE_DIR}/{DATASET_DIR}/SEG_{SEGMENT_LENGTH}_OVERLAP_{int(OVERLAP_SIZE*100)}_AUG_{AUGMENT_SIZE}"

# %%
if SEGMENT_TEST and not os.path.exists(f"{BASE_DIR}/test"):
    print("SEG TEST")
    segment_dataset(
        f"{ANNOTATE_DIR}/{DATASET_DIR}",
        f"{BASE_DIR}/test",
        base_trans=[*BASE_TRANSFORM, *TRAIN_TRANSFORM],
        overlap_size=OVERLAP_SIZE,
        segment_length=SEGMENT_LENGTH,
        plot_distribution=True,
        base_per_segment=True,
    )

# exit()
# %%

if SEGMENT_VALID and not os.path.exists(f"{BASE_DIR}/valid"):
    print("SEG VALID")
    segment_dataset(
        f"{ANNOTATE_DIR}/{DATASET_DIR}",
        f"{BASE_DIR}/valid",
        base_trans=[*BASE_TRANSFORM, *TRAIN_TRANSFORM],
        overlap_size=OVERLAP_SIZE,
        segment_length=SEGMENT_LENGTH,
        plot_distribution=True,
        base_per_segment=True,
    )

# %%
if SEGMENT_TRAIN and not os.path.exists(f"{BASE_DIR}/train"):
    print("SEG TRAIN")
    segment_dataset(
        f"{ANNOTATE_DIR}/{DATASET_DIR}",
        f"{BASE_DIR}/train",
        base_trans=BASE_TRANSFORM,
        extra_trans=TRAIN_TRANSFORM,
        overlap_size=OVERLAP_SIZE,
        augment_size=AUGMENT_SIZE,
        segment_length=SEGMENT_LENGTH,
        plot_distribution=True,
        extra_per_segment=False,
    )
# exit()
# %% REPRESENTATION
if REPRESENT_TEST and not os.path.exists(f"{BASE_DIR}/MFCC_{MFCC_COEFF}/test"):
    print("REP TEST")
    mat_dict_test = represent_dataset(
        f"{BASE_DIR}/test",
        f"{BASE_DIR}/MFCC_{MFCC_COEFF}/test",
        n_mfcc=MFCC_COEFF,
        n_fft=MFCC_N_FFT,
        hop_length=MFCC_HOP_LENGTH,
    )
# %%
if REPRESENT_VALID and not os.path.exists(f"{BASE_DIR}/MFCC_{MFCC_COEFF}/valid"):
    print("REP VALID")

    mat_dict_valid = represent_dataset(
        f"{BASE_DIR}/valid",
        f"{BASE_DIR}/MFCC_{MFCC_COEFF}/valid",
        n_mfcc=MFCC_COEFF,
        n_fft=MFCC_N_FFT,
        hop_length=MFCC_HOP_LENGTH,
    )
# %%
if REPRESENT_TRAIN and not os.path.exists(f"{BASE_DIR}/MFCC_{MFCC_COEFF}/train"):
    print("REP TRAIN")

    mat_dict_train = represent_dataset(
        f"{BASE_DIR}/train",
        f"{BASE_DIR}/MFCC_{MFCC_COEFF}/train",
        n_mfcc=MFCC_COEFF,
        n_fft=MFCC_N_FFT,
        hop_length=MFCC_HOP_LENGTH,
    )

# %%
# exit()
# %% LOAD REPRESENTATION
if not REPRESENT_TEST or os.path.exists(f"{BASE_DIR}/MFCC_{MFCC_COEFF}/test"):
    mat_dict_test = load_mat_representation(
        f"{BASE_DIR}/MFCC_{MFCC_COEFF}/test/representation.mat"
    )

if not REPRESENT_TRAIN or os.path.exists(f"{BASE_DIR}/MFCC_{MFCC_COEFF}/train"):
    mat_dict_train = load_mat_representation(
        f"{BASE_DIR}/MFCC_{MFCC_COEFF}/train/representation.mat"
    )

if not REPRESENT_VALID or os.path.exists(f"{BASE_DIR}/MFCC_{MFCC_COEFF}/valid"):
    mat_dict_valid = load_mat_representation(
        f"{BASE_DIR}/MFCC_{MFCC_COEFF}/valid/representation.mat"
    )

# exit()
# %% NP.ARRAY
unique_labels = list(set(mat_dict_train["label"]))

X_train = np.array(mat_dict_train["representation"])
y_train = np.array(mat_dict_train["label"])

X_valid = np.array(mat_dict_valid["representation"])
y_valid = np.array(mat_dict_valid["label"])

X_test = np.array(mat_dict_test["representation"])
y_test = np.array(mat_dict_test["label"])
# %% SCALER
se = StandardScaler()

X_train_rep = tf.convert_to_tensor(
    se.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
)
X_valid_rep = tf.convert_to_tensor(
    se.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
)
X_test_rep = tf.convert_to_tensor(
    se.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
)
# X_train_rep = tf.convert_to_tensor(X_train)
# X_test_rep = tf.convert_to_tensor(X_test)
# X_valid_rep = tf.convert_to_tensor(X_valid)

# print(X_train_rep.shape)
# print(X_valid_rep.shape)
# print(X_test_rep.shape)
# print(y_train.shape)
# print(y_valid.shape)
# print(y_test.shape)
# %% BUILD MODEL
model = build_perceptron(
    output_size=len(unique_labels),
    shape_size=X_train_rep.shape,
    dense1=MODEL_DENSE_1,
    dropout1=MODEL_DROPOUT_1,
    dense2=MODEL_DENSE_2,
    dropout2=MODEL_DROPOUT_2,
    dense3=MODEL_DENSE_3,
    learning_rate=LEARNING_RATE,
)

# %% MODEL SUMMARY
model_arch = model.to_json()
# model.summary()
# exit()

# %% TRAIN MODEL
history, time_history = train_model(
    model,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    patience=PATIENCE,
    X_train=X_train_rep,
    y_train=y_train,
    X_validation=X_valid_rep,
    y_validation=y_valid,
    verbose=2,
)

# print(time_history.times)

# exit()
# %%
test_loss, test_acc = model.evaluate(X_test_rep, y_test, verbose=2)

# %%
y_pred = model.predict(X_test_rep)
y_pred = np.argmax(y_pred, axis=1)

f1_micro = f1_score(y_test, y_pred, average="micro")
f1_macro = f1_score(y_test, y_pred, average="macro")

confusion = tf.math.confusion_matrix(y_pred, y_test)

# %% SAVING PROCESS
timestamp = int(time.time())

save_foldername = f"{MODELS_DIR}/{DATASET_DIR}/SEG_{SEGMENT_LENGTH}_OVERLAP_{int(OVERLAP_SIZE*100)}_AUG_{AUGMENT_SIZE}/MFCC_{MFCC_COEFF}/D{MODEL_DENSE_1}_DO{MODEL_DROPOUT_1}_D{MODEL_DENSE_2}_DO{MODEL_DROPOUT_2}_D{MODEL_DENSE_3}/{timestamp}_{test_acc * 100}"

utils.create_dir_hierarchy(save_foldername)

with open(f"{save_foldername}/model_architecture.json", "w") as f:
    f.write(model_arch)

model.save(f"{save_foldername}/model.h5")

pkl.dump(se, open(f"{save_foldername}/scaler.pkl", "wb"))

plot_history(history, save_path=f"{save_foldername}")

plot_confusion_matrix(
    confusion.numpy(), size=len(unique_labels), save_path=f"{save_foldername}"
)

test_labels, test_count = np.unique(y_test, return_counts=True)
plot_class_distribution(
    test_labels,
    test_count,
    save_path=f"{save_foldername}",
    filename="test_distribution.jpg",
)

valid_labels, valid_count = np.unique(y_valid, return_counts=True)
plot_class_distribution(
    valid_labels,
    valid_count,
    save_path=f"{save_foldername}",
    filename="valid_distribution.jpg",
)

train_labels, train_count = np.unique(y_train, return_counts=True)
plot_class_distribution(
    train_labels,
    train_count,
    save_path=f"{save_foldername}",
    filename="train_distribution.jpg",
)


# %%
overview = {
    "dataset_dir": DATASET_DIR,
    "classes": len(unique_labels),
    "segment_length": SEGMENT_LENGTH,
    "sample_rate": SAMPLE_RATE,
    "augment_size": AUGMENT_SIZE,
    "overlap_size": OVERLAP_SIZE,
    "train_shape": X_train_rep.numpy().shape,
    "valid_shape": X_valid_rep.numpy().shape,
    "test_shape": X_test_rep.numpy().shape,
    "train_time": sum(time_history.times),
    "scores": {
        "test_macro": f1_macro,
        "test_micro": f1_micro,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_loss": history.history["loss"][-1],
        "train_acc": history.history["accuracy"][-1],
        "valid_loss": history.history["val_loss"][-1],
        "valid_acc": history.history["val_accuracy"][-1],
    },
    "training_params": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
    },
    "representation": {
        "name": "MFCC",
        "n_mfcc": MFCC_COEFF,
        "n_fft": MFCC_N_FFT,
        "hop_length": MFCC_HOP_LENGTH,
    },
}

# %%
with open(f"{save_foldername}/overview.json", "w") as f:
    f.write(json.dumps(overview))
