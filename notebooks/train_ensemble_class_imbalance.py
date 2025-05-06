import os
import argparse
import pickle
import json
from datetime import date
import random
import time

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
from sklearn.cluster import DBSCAN
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def load_patches_from_dir(input_dir):
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
    return np.array(patches), np.array(labels), patch_files


def filter_black(data, mask_limit=0.1):
    masked_fraction = np.array([
        np.sum(np.mean(patch, axis=-1) < 10) / patch.shape[0]**2 for patch in data
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

    patches = patches.astype("float32")
    patches[:, :, :, :num_s2_bands] = np.clip(patches[:, :, :, :num_s2_bands] / 10000, 0, 1)

    return patches, labels, train_indices, val_indices, test_indices


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
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )
    return model


def main(input_dir, output_dir, resolution):
    os.makedirs(output_dir, exist_ok=True)

    patches, labels, patch_files = load_patches_from_dir(input_dir)
    pos_patches = patches[labels == 1]
    neg_patches = patches[labels == 0]
    pos_filtered = filter_black(pos_patches, mask_limit=0.1)
    neg_filtered = filter_black(neg_patches, mask_limit=0.6)

    x_all = np.concatenate([pos_filtered, neg_filtered])
    y_all = np.concatenate([np.ones(len(pos_filtered)), np.zeros(len(neg_filtered))])
    pos_centers = [(i, i) for i in range(len(pos_filtered))]
    neg_centers = [(i, i) for i in range(len(neg_filtered))]

    patches, labels, train_idx_base, val_idx_base, test_idx = spatial_split(
        x_all, y_all, pos_centers, neg_centers)

    combined_train_val = train_idx_base + val_idx_base
    test_labels = labels[test_idx]
    label_counts_test = {
        "positive": int(np.sum(test_labels)),
        "negative": int(len(test_labels) - np.sum(test_labels))
    }

    input_shape = patches.shape[1:]
    batch_size = 16
    epochs = 160
    class_weight = {
        0: 1.48,
        1: 3.07
    }

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
    'fill_mode': 'reflect'
    }

    datagen = ImageDataGenerator(**augmentation_parameters)

    all_preds = []
    label_counts_per_model = []
    split_indices_per_model = []
    start_time = time.time()

    x_test = patches[test_idx]
    for seed in [1, 2, 3, 4, 5]:
        random.seed(seed)
        shuffled = combined_train_val.copy()
        random.shuffle(shuffled)
        n_train = int(0.82 * len(shuffled))
        this_train_idx = shuffled[:n_train]
        this_val_idx = shuffled[n_train:]
        x_train, y_train = patches[this_train_idx], labels[this_train_idx]
        x_val, y_val = patches[this_val_idx], labels[this_val_idx]

        keras.utils.set_random_seed(seed)
        model = build_model(input_shape)
        history = model.fit(datagen.flow(x_train, y_train),
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            shuffle=True,
                            class_weight=class_weight,
                            verbose=1)
        preds = model(x_test, training=False).numpy().flatten()
        all_preds.append(preds)

        label_counts_per_model.append({
            "seed": seed,
            "train": {"positive": int(np.sum(y_train)), "negative": int(len(y_train) - np.sum(y_train))},
            "val": {"positive": int(np.sum(y_val)), "negative": int(len(y_val) - np.sum(y_val))}
        })
        split_indices_per_model.append({
            "seed": seed,
            "train_idx": this_train_idx,
            "val_idx": this_val_idx
        })

    training_duration = round(time.time() - start_time, 2)
    ensemble_preds = np.mean(all_preds, axis=0)
    binary_preds = ensemble_preds > 0.5

    report = classification_report(test_labels, binary_preds, output_dict=True)
    conf_matrix = confusion_matrix(test_labels, binary_preds).tolist()
    auc_roc = roc_auc_score(test_labels, ensemble_preds)
    auc_pr = average_precision_score(test_labels, ensemble_preds)
    fpr, tpr, _ = roc_curve(test_labels, ensemble_preds)
    precision, recall, _ = precision_recall_curve(test_labels, ensemble_preds)

    version = f"{resolution}px_ensemble_{date.today().isoformat()}"

    np.save(os.path.join(output_dir, version + '_preds.npy'), ensemble_preds)
    np.save(os.path.join(output_dir, version + '_y_test.npy'), test_labels)
    np.savez(os.path.join(output_dir, version + '_curves.npz'),
             fpr=fpr, tpr=tpr, precision=precision, recall=recall)
    with open(os.path.join(output_dir, version + '_last_model_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    with open(os.path.join(output_dir, version + '_model_config.json'), 'w') as f:
        f.write(model.to_json())
    with open(os.path.join(output_dir, version + '_split_indices_per_model.json'), 'w') as f:
        json.dump(split_indices_per_model, f, indent=4)

    results = {
        "version": version,
        "patch_resolution": resolution,
        "batch_size": batch_size,
        "epochs": epochs,
        "class_weights": class_weight,
        "augmentation_parameters": {k: str(v) for k, v in augmentation_parameters.items()},
        "training_duration": training_duration,
        "label_counts": {
            "test": label_counts_test,
            "per_model_train_val": label_counts_per_model
        },
        "metrics": {
            "roc_auc": auc_roc,
            "pr_auc": auc_pr,
            "classification_report": report,
            "confusion_matrix": conf_matrix
        }
    }

    with open(os.path.join(output_dir, version + '_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, version + '_roc_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f'PR Curve (AUC = {auc_pr:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, version + '_pr_curve.png'))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, required=True)
    parser.add_argument('--output_directory', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=256)
    args = parser.parse_args()
    main(args.input_directory, args.output_directory, args.resolution)