import os
import sys
import argparse
import pickle
from datetime import date
import random

import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.cluster import DBSCAN
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_patches_from_dir(input_dir, resolution):
    patch_files = [f for f in os.listdir(input_dir) if f.endswith('patch_arrays.pkl')]
    label_files = [f.replace('patch_arrays.pkl', 'patch_array_labels.pkl') for f in patch_files]

    patches, labels = [], []

    for patch_file, label_file in zip(patch_files, label_files):
        with open(os.path.join(input_dir, patch_file), 'rb') as f:
            patch_data = pickle.load(f)
            patches.extend(patch_data)
        with open(os.path.join(input_dir, label_file), 'rb') as f:
            label_data = pickle.load(f)
            labels = np.concatenate((labels, label_data)) if len(labels) else label_data

    patches = np.array(patches)
    labels = np.array(labels)
    return patches, labels, patch_files

def filter_black(data, mask_limit=0.1):
    masked_fraction = np.array([
        np.sum(np.mean(patch, axis=-1) < 10) / np.size(np.mean(patch, axis=-1)) for patch in data
    ])
    return data[masked_fraction < mask_limit]

def spatial_split(patches, labels, positive_centers, negative_centers, num_s2_bands=12):
    all_centers = np.array(positive_centers + negative_centers)
    cluster_labels = DBSCAN(eps=0.1, min_samples=2).fit_predict(all_centers)

    unique_clusters = np.unique(cluster_labels)
    random.seed(7)
    random.shuffle(unique_clusters)

    n = len(unique_clusters)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_clusters = set(unique_clusters[:n_train])
    val_clusters = set(unique_clusters[n_train:n_train + n_val])
    test_clusters = set(unique_clusters[n_train + n_val:])

    train_indices = [i for i, c in enumerate(cluster_labels) if c in train_clusters]
    val_indices = [i for i, c in enumerate(cluster_labels) if c in val_clusters]
    test_indices = [i for i, c in enumerate(cluster_labels) if c in test_clusters]

    # Normalize only the Sentinel-2 bands
    patches = patches.astype("float32")
    patches[:, :, :, :num_s2_bands] = np.clip(patches[:, :, :, :num_s2_bands] / 10000, 0, 1)

    x_train, y_train = patches[train_indices], labels[train_indices]
    x_val, y_val = patches[val_indices], labels[val_indices]
    x_test, y_test = patches[test_indices], labels[test_indices]

    return x_train, y_train, x_val, y_val, x_test, y_test

def build_model(input_shape):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding='same', activation="relu"),
        layers.Conv2D(32, 3, padding='same', activation="relu"),
        layers.Conv2D(32, 3, padding='same', activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding='same', activation="relu"),
        layers.Conv2D(32, 3, padding='same', activation="relu"),
        layers.Conv2D(32, 3, padding='same', activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding='same', activation="relu"),
        layers.Conv2D(32, 3, padding='same', activation="relu"),
        layers.Conv2D(32, 3, padding='same', activation="relu"),
        layers.MaxPooling2D(3),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="acc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")]
    )
    return model

def main(input_dir, output_dir, resolution):
    keras.utils.set_random_seed(42)

    # Load patches
    patches, labels, patch_files = load_patches_from_dir(input_dir, resolution)
    positive_patches = patches[labels == 1]
    negative_patches = patches[labels == 0]

    filtered_positives = filter_black(positive_patches, mask_limit=0.1)
    filtered_negatives = filter_black(negative_patches, mask_limit=0.6)


    

    # Fake centers for spatial split (replace with real tile centers if available)
    positive_centers = [(i, i) for i in range(len(filtered_positives))]
    negative_centers = [(i, i) for i in range(len(filtered_negatives))]

    x_train, y_train, x_val, y_val, x_test, y_test = spatial_split(
        np.concatenate([filtered_positives, filtered_negatives]), 
        np.concatenate([np.ones(len(filtered_positives)), np.zeros(len(filtered_negatives))]),
        positive_centers, negative_centers
    )
    
    np.savez(os.path.join(output_dir, "split_indices.npz"), train_idx=train_indices, val_idx=val_indices, test_idx=test_indices)

    input_shape = x_train.shape[1:]
    model = build_model(input_shape)

    augmentation_parameters = {
    'featurewise_center': False,
    'rotation_range': 360,
    'width_shift_range': [0.9, 1.1],
    'height_shift_range': [0.9, 1.1],
    'shear_range': 10,
    'zoom_range': [0.9, 1.1],
    'vertical_flip': True,
    'horizontal_flip': True,
    # Fill options: "constant", "nearest", "reflect" or "wrap"
    'fill_mode': 'reflect'}
    
    datagen = ImageDataGenerator(**augmentation_parameters)

    batch_size = 32
    epochs = 160
    class_weight = {0: 1, 1: 1}

    history = model.fit(datagen.flow(x_train, y_train),
                        batch_size=batch_size, epochs=epochs,
                        validation_data=(x_val, y_val),
                        shuffle=True,
                        class_weight=class_weight)

    # Evaluation
    threshold = 0.8
    preds = model.predict(x_test)
    report = classification_report(y_test, preds > threshold, target_names=['No Mine', 'Mine'])

    # Save
    version = f"{resolution}px_{date.today().isoformat()}"
    os.makedirs(output_dir, exist_ok=True)

    model.save(os.path.join(output_dir, version + '.h5'))

    with open(os.path.join(output_dir, version + '_config.txt'), 'w') as f:
        f.write(f"Model version: {version}\n\n")
        f.write('Input Data:\n')
        [f.write('\t' + file + '\n') for file in patch_files]
        f.write('\n\nAugmentation Parameters:\n')
        for k, v in zip(augmentation_parameters.keys(), augmentation_parameters.values()):
                f.write(f"\t{k}: {v}\n")
        f.write(f"\nBatch Size: {batch_size}")
        f.write(f"\nTraining Epochs: {epochs}")
        f.write(f"\nClass Weights: {class_weight}")
        f.write(f'\n\nClassification Report at {threshold}\n')
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, required=True, help="Folder with 4 .pkl files")
    parser.add_argument('--output_directory', type=str, required=True, help="Folder to save model + config")
    parser.add_argument('--resolution', type=int, default=256, help="Patch resolution")
    args = parser.parse_args()
    main(args.input_directory, args.output_directory, args.resolution)