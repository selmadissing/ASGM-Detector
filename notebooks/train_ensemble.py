import os
import argparse
import pickle
import json
from datetime import date
import random
import time
import geopandas as gpd

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, f1_score
)
from sklearn.cluster import DBSCAN
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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


def filter_black(data, mask_limit=0.1, return_mask=False):
    masked_fraction = np.array([
        np.sum(np.mean(patch, axis=-1) < 10) / patch.shape[0]**2 for patch in data
    ])
    mask = masked_fraction < mask_limit
    if return_mask:
        return data[mask], mask
    return data[mask]

def find_geojson_file(directory, suffix):
    matches = [f for f in os.listdir(directory) if f.endswith(suffix)]
    if len(matches) == 0:
        raise FileNotFoundError(f"No GeoJSON file found ending with '{suffix}' in {directory}")
    elif len(matches) > 1:
        raise ValueError(f"Multiple files found ending with '{suffix}' in {directory}: {matches}")
    return os.path.join(directory, matches[0])


def spatial_split(patches, labels, positive_centers, negative_centers, num_s2_bands=12):
    all_centers = np.array(positive_centers + negative_centers)
    cluster_labels = DBSCAN(eps=3000, min_samples=1).fit_predict(all_centers)

    if np.any(cluster_labels == -1):
        print(f"⚠️ Warning: {np.sum(cluster_labels == -1)} patches marked as noise and excluded from splitting.")

    unique_clusters = np.unique(cluster_labels)
    random.shuffle(unique_clusters)

    n = len(unique_clusters)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

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
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ]
    )
    return model


def plot_history(history, seed, output_dir, version):
    metrics = ['loss', 'precision', 'recall', 'auc']
    for metric in metrics:
        val_metric = 'val_' + metric
        if metric in history.history and val_metric in history.history:
            plt.figure()
            plt.plot(history.history[metric], label=f'Train {metric}')
            plt.plot(history.history[val_metric], label=f'Val {metric}')
            plt.title(f'Training vs. Validation {metric.capitalize()} (Seed {seed})')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            filename = os.path.join(output_dir, f"{version}_seed{seed}_{metric}.png")
            plt.savefig(filename)
            plt.close()





def main(input_dir, output_dir, experiment_name):
    os.makedirs(output_dir, exist_ok=True)

    version = f"{experiment_name}px_{date.today().isoformat()}" 

    patches, labels, patch_files = load_patches_from_dir(input_dir)
    pos_patches = patches[labels == 1]
    neg_patches = patches[labels == 0]
    pos_filtered, pos_mask = filter_black(pos_patches, mask_limit=0.1, return_mask=True)
    neg_filtered, neg_mask = filter_black(neg_patches, mask_limit=0.6, return_mask=True)

    x_all = np.concatenate([pos_filtered, neg_filtered])
    y_all = np.concatenate([np.ones(len(pos_filtered)), np.zeros(len(neg_filtered))])

    # Find and load positive and negative sampling GeoJSONs
    positive_geojson_path = find_geojson_file(input_dir, "_positives.geojson")
    negative_geojson_path = find_geojson_file(input_dir, "_negatives.geojson")

    pos_gdf = gpd.read_file(positive_geojson_path).to_crs(epsg=3857)
    neg_gdf = gpd.read_file(negative_geojson_path).to_crs(epsg=3857)

    # Clean invalid or empty geometries
    pos_gdf = pos_gdf[pos_gdf.geometry.is_valid & ~pos_gdf.geometry.is_empty]
    neg_gdf = neg_gdf[neg_gdf.geometry.is_valid & ~neg_gdf.geometry.is_empty]

    pos_gdf = pos_gdf[pos_mask].reset_index(drop=True)
    neg_gdf = neg_gdf[neg_mask].reset_index(drop=True)

    # Validate alignment between .geojson and .pkl patches
    assert len(pos_filtered) == len(pos_gdf), (
        f"Mismatch: {len(pos_filtered)} filtered patches vs {len(pos_gdf)} geometries in {positive_geojson_path}"
    )
    assert len(neg_filtered) == len(neg_gdf), (
        f"Mismatch: {len(neg_filtered)} filtered patches vs {len(neg_gdf)} geometries in {negative_geojson_path}"
    )

    print(f"✅ Positives: kept {len(pos_filtered)} / {len(pos_patches)}")
    print(f"✅ Negatives: kept {len(neg_filtered)} / {len(neg_patches)}")

    # Extract actual spatial coordinates for DBSCAN
    pos_centers = [(geom.x, geom.y) for geom in pos_gdf.geometry]
    neg_centers = [(geom.x, geom.y) for geom in neg_gdf.geometry]

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
    class_weight = {0: 1, 1: 1}

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
    best_thresholds = []
    start_time = time.time()

    x_test = patches[test_idx]
    np.save(os.path.join(output_dir, version + '_x_test_patches.npy'), x_test)
    np.save(os.path.join(output_dir, version + '_y_test.npy'), test_labels)
    for seed in [1, 2, 3, 4, 5]:
        random.seed(seed)
        shuffled = combined_train_val.copy()
        this_train_idx, this_val_idx = train_test_split(
            shuffled, test_size=0.3, stratify=labels[shuffled], random_state=seed
        )

        x_train, y_train = patches[this_train_idx], labels[this_train_idx]
        x_val, y_val = patches[this_val_idx], labels[this_val_idx]

        label_counts_per_model.append({
            "seed": seed,
            "train": {"positive": int(np.sum(y_train)), "negative": int(len(y_train) - np.sum(y_train))},
            "val": {"positive": int(np.sum(y_val)), "negative": int(len(y_val) - np.sum(y_val))}
        })

        keras.utils.set_random_seed(seed)
        model = build_model(input_shape)
        
        history = model.fit(datagen.flow(x_train, y_train),
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            shuffle=True,
                            class_weight=class_weight,
                            verbose=1)
        plot_history(history, seed, output_dir, version)
        
        model_save_path = os.path.join(output_dir, f"model_seed{seed}.keras")
        model.save(model_save_path)

        # Get validation predictions
        val_preds = model(x_val, training=False).numpy().flatten()

        # Find threshold that maximizes F1
        precision, recall, thresholds = precision_recall_curve(y_val, val_preds)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        print(f"Best threshold for seed {seed}: {best_threshold:.4f} (F1 = {f1_scores[best_idx]:.4f})")
        best_thresholds.append({"seed": seed, "threshold": best_threshold})

        # Get test predictions
        preds = model(x_test, training=False).numpy().flatten()
        all_preds.append(preds)

        

    training_duration = round(time.time() - start_time, 2)
    ensemble_preds = np.mean(all_preds, axis=0)
    binary_preds = ensemble_preds > 0.5   # LOOK INTO THIS

    report = classification_report(test_labels, binary_preds, output_dict=True)
    conf_matrix = confusion_matrix(test_labels, binary_preds).tolist()
    auc_roc = roc_auc_score(test_labels, ensemble_preds)
    auc_pr = average_precision_score(test_labels, ensemble_preds)
    fpr, tpr, _ = roc_curve(test_labels, ensemble_preds)
    precision, recall, _ = precision_recall_curve(test_labels, ensemble_preds)


    np.save(os.path.join(output_dir, version + '_preds.npy'), ensemble_preds)
    np.savez(os.path.join(output_dir, version + '_curves.npz'),
             fpr=fpr, tpr=tpr, precision=precision, recall=recall)
    with open(os.path.join(output_dir, version + '_model_config.json'), 'w') as f:
        f.write(model.to_json())

    results = {
        "version": version,
        "input_shape": input_shape,
        "batch_size": batch_size,
        "epochs": epochs,
        "class_weights": class_weight,
        "augmentation_parameters": augmentation_parameters,
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
    parser.add_argument('--experiment_name', type=str, required=True)
    args = parser.parse_args()
    main(args.input_directory, args.output_directory, args.experiment_name)