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
from sklearn.metrics import precision_score, recall_score

# Load patch arrays and their labels from a directory
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
    return np.array(patches), np.array(labels)

# Filter out patches that are mostly black (low mean pixel value)
def filter_black(data, mask_limit=0.1, return_mask=False):
    masked_fraction = np.array([
        np.sum(np.mean(patch, axis=-1) < 10) / patch.shape[0]**2 for patch in data
    ])
    mask = masked_fraction < mask_limit
    if return_mask:
        return data[mask], mask
    return data[mask]

# Find a geojson file in a directory with a specific suffix
def find_geojson_file(directory, suffix):
    matches = [f for f in os.listdir(directory) if f.endswith(suffix)]
    if len(matches) == 0:
        raise FileNotFoundError(f"No GeoJSON file found ending with '{suffix}' in {directory}")
    elif len(matches) > 1:
        raise ValueError(f"Multiple files found ending with '{suffix}' in {directory}: {matches}")
    return os.path.join(directory, matches[0])

# Split data spatially into train, validation, and test sets using DBSCAN clustering
def spatial_split(patches, labels, positive_centers, negative_centers, num_s2_bands=12, seed=7):
    all_centers = np.array(positive_centers + negative_centers)
    cluster_labels = DBSCAN(eps=3000, min_samples=1).fit_predict(all_centers)

    if np.any(cluster_labels == -1):
        print(f"⚠️ Warning: {np.sum(cluster_labels == -1)} patches marked as noise and excluded from splitting.")

    unique_clusters = np.unique(cluster_labels)

    # Split clusters into train, val, and test
    train_clusters, temp_clusters = train_test_split(unique_clusters, train_size=0.7, random_state=seed)
    val_clusters, test_clusters = train_test_split(temp_clusters, test_size=0.5, random_state=seed)

    train_indices = [i for i, c in enumerate(cluster_labels) if c in train_clusters]
    val_indices = [i for i, c in enumerate(cluster_labels) if c in val_clusters]
    test_indices = [i for i, c in enumerate(cluster_labels) if c in test_clusters]

    # Normalize Sentinel-2 bands
    patches = patches.astype("float32")
    patches[:, :, :, :num_s2_bands] = np.clip(patches[:, :, :, :num_s2_bands] / 10000, 0, 1)

    return patches, labels, train_indices, val_indices, test_indices

# Build a CNN model for binary classification
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

# Plot training history (loss, precision, recall, auc) and save plots
def plot_history(history, output_dir):
    metrics = ['loss', 'precision', 'recall', 'auc']
    for metric in metrics:
        val_metric = 'val_' + metric
        if metric in history.history and val_metric in history.history:
            plt.figure()
            plt.plot(history.history[metric], label=f'Train {metric}')
            plt.plot(history.history[val_metric], label=f'Val {metric}')
            plt.title(f'Training vs. Validation {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            filename = os.path.join(output_dir, f"{metric}.png")
            plt.savefig(filename)
            plt.close()

# Compute bootstrap confidence intervals for F1, precision, and recall
def bootstrap_metrics(y_true, y_pred, n_iterations=1000, seed=42):
    rng = np.random.default_rng(seed)
    f1_scores = []
    precisions = []
    recalls = []

    for _ in range(n_iterations):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_sample = y_true[indices]
        y_pred_sample = y_pred[indices]

        f1_scores.append(f1_score(y_sample, y_pred_sample))
        precisions.append(precision_score(y_sample, y_pred_sample))
        recalls.append(recall_score(y_sample, y_pred_sample))

    ci = lambda scores: (np.percentile(scores, 2.5), np.percentile(scores, 97.5))

    return {
        "f1": (np.mean(f1_scores), *ci(f1_scores)),
        "precision": (np.mean(precisions), *ci(precisions)),
        "recall": (np.mean(recalls), *ci(recalls))
    }

# Main training and evaluation pipeline
def main(input_dir, output_dir, experiment_name):
    SEED = 7
    random.seed(SEED)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    os.makedirs(output_dir, exist_ok=True)
    version = f"{experiment_name}_{date.today().isoformat()}"

    # Load and filter patches
    patches, labels = load_patches_from_dir(input_dir)
    pos_patches = patches[labels == 1]
    neg_patches = patches[labels == 0]
    pos_filtered, pos_mask = filter_black(pos_patches, mask_limit=0.1, return_mask=True)
    neg_filtered, neg_mask = filter_black(neg_patches, mask_limit=0.6, return_mask=True)

    x_all = np.concatenate([pos_filtered, neg_filtered])
    y_all = np.concatenate([np.ones(len(pos_filtered)), np.zeros(len(neg_filtered))])

    # Load geojsons for spatial splitting
    positive_geojson_path = find_geojson_file(input_dir, "_positives.geojson")
    negative_geojson_path = find_geojson_file(input_dir, "_negatives.geojson")

    pos_gdf = gpd.read_file(positive_geojson_path).to_crs(epsg=3857)
    neg_gdf = gpd.read_file(negative_geojson_path).to_crs(epsg=3857)

    pos_gdf = pos_gdf[pos_gdf.geometry.is_valid & ~pos_gdf.geometry.is_empty]
    neg_gdf = neg_gdf[neg_gdf.geometry.is_valid & ~neg_gdf.geometry.is_empty]
    pos_gdf = pos_gdf[pos_mask].reset_index(drop=True)
    neg_gdf = neg_gdf[neg_mask].reset_index(drop=True)

    assert len(pos_filtered) == len(pos_gdf)
    assert len(neg_filtered) == len(neg_gdf)

    print(f"✅ Positives: kept {len(pos_filtered)} / {len(pos_patches)}")
    print(f"✅ Negatives: kept {len(neg_filtered)} / {len(neg_patches)}")

    # Get patch centers for clustering
    pos_centers = [(geom.x, geom.y) for geom in pos_gdf.geometry]
    neg_centers = [(geom.x, geom.y) for geom in neg_gdf.geometry]

    # Spatial split
    patches, labels, train_idx, val_idx, test_idx = spatial_split(
        x_all, y_all, pos_centers, neg_centers, seed=SEED)

    test_labels = labels[test_idx]
    label_counts_test = {
        "positive": int(np.sum(test_labels)),
        "negative": int(len(test_labels) - np.sum(test_labels))
    }

    input_shape = patches.shape[1:]
    batch_size = 16
    epochs = 160

    # Data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,
        rotation_range=360,
        width_shift_range=[0.9, 1.1],
        height_shift_range=[0.9, 1.1],
        shear_range=10,
        zoom_range=[0.9, 1.1],
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='reflect'
    )

    start_time = time.time()

    # Save test set for later use
    np.save(os.path.join(output_dir, 'x_test_patches.npy'), patches[test_idx])
    np.save(os.path.join(output_dir, 'y_test.npy'), test_labels)

    x_train, y_train = patches[train_idx], labels[train_idx]
    x_val, y_val = patches[val_idx], labels[val_idx]

    train_counts = {
        "positive": int(np.sum(y_train)),
        "negative": int(len(y_train) - np.sum(y_train))}
    val_counts = {
        "positive": int(np.sum(y_val)),
        "negative": int(len(y_val) - np.sum(y_val))}

    print(f"Training set: {train_counts['positive']} positives, {train_counts['negative']} negatives")
    print(f"Validation set: {val_counts['positive']} positives, {val_counts['negative']} negatives")
    print(f"Test set: {label_counts_test['positive']} positives, {label_counts_test['negative']} negatives")

    # Compute class weights for imbalanced data
    train_total = len(y_train)
    n_positive = np.sum(y_train)
    n_negative = train_total - n_positive
    class_weight = {
        0: train_total / (2 * n_negative),  # weight for negative class
        1: train_total / (2 * n_positive)   # weight for positive class
    }
    print(f"✅ Computed class weights: {class_weight}")

    # Build and train model
    model = build_model(input_shape)

    history = model.fit(datagen.flow(x_train, y_train),
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        shuffle=True,
                        class_weight=class_weight,
                        verbose=1)

    # Plot and save training history
    plot_history(history, output_dir)
    model.save(os.path.join(output_dir, 'model.keras'))

    # Find best threshold on validation set using F1 score
    val_preds = model(x_val, training=False).numpy().flatten()
    precision, recall, thresholds = precision_recall_curve(y_val, val_preds)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"Best threshold based on validation set: {best_threshold:.4f} (F1 = {f1_scores[best_idx]:.4f})")

    # Evaluate on test set
    preds_test = model(patches[test_idx], training=False).numpy().flatten()
    binary_preds = preds_test > best_threshold

    auc_roc = roc_auc_score(test_labels, preds_test)
    auc_pr = average_precision_score(test_labels, preds_test)
    fpr, tpr, _ = roc_curve(test_labels, preds_test)
    test_precision, test_recall, _ = precision_recall_curve(test_labels, preds_test)
    report = classification_report(test_labels, binary_preds, output_dict=True)
    conf_matrix = confusion_matrix(test_labels, binary_preds).tolist()

    # Bootstrap confidence intervals
    bootstrap_results = bootstrap_metrics(test_labels, binary_preds)

    # Print results
    print(f"Test F1: {bootstrap_results['f1'][0]:.3f} (95% CI: [{bootstrap_results['f1'][1]:.3f}, {bootstrap_results['f1'][2]:.3f}])")
    print(f"Test Precision: {bootstrap_results['precision'][0]:.3f} (95% CI: [{bootstrap_results['precision'][1]:.3f}, {bootstrap_results['precision'][2]:.3f}])")
    print(f"Test Recall: {bootstrap_results['recall'][0]:.3f} (95% CI: [{bootstrap_results['recall'][1]:.3f}, {bootstrap_results['recall'][2]:.3f}])")

    training_duration = round(time.time() - start_time, 2)

    # Save predictions and curves
    np.save(os.path.join(output_dir, 'preds.npy'), preds_test)
    np.savez(os.path.join(output_dir, 'curves.npz'),
             fpr=fpr, tpr=tpr, precision=test_precision, recall=test_recall)
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        f.write(model.to_json())

    # Prepare results for saving
    results = {
        "version": version,
        "input_shape": input_shape,
        "batch_size": batch_size,
        "epochs": epochs,
        "class_weights": class_weight,
        "augmentation_parameters": datagen.__dict__,
        "training_duration": training_duration,
        "label_counts": {
            "test": label_counts_test,
            "train": train_counts, 
            "val": val_counts
        },
        "best_threshold": best_threshold,
        "best_f1_score": f1_scores[best_idx],
        "metrics": {
            "roc_auc": auc_roc,
            "pr_auc": auc_pr,
            "f1_score": f1_score(test_labels, binary_preds),
            "classification_report": report,
            "confusion_matrix": conf_matrix
        },
        "bootstrap_confidence_intervals": {
            "f1": {
                "mean": bootstrap_results['f1'][0],
                "ci_lower": bootstrap_results['f1'][1],
                "ci_upper": bootstrap_results['f1'][2]
            },
            "precision": {
                "mean": bootstrap_results['precision'][0],
                "ci_lower": bootstrap_results['precision'][1],
                "ci_upper": bootstrap_results['precision'][2]
            },
            "recall": {
                "mean": bootstrap_results['recall'][0],
                "ci_lower": bootstrap_results['recall'][1],
                "ci_upper": bootstrap_results['recall'][2]
            }
        }
    }

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(convert_numpy(results), f, indent=4)

    # Plot and save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # Plot and save Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, label=f'PR Curve (AUC = {auc_pr:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()

# Entry point: parse arguments and run main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, required=True)
    parser.add_argument('--output_directory', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    args = parser.parse_args()
    main(args.input_directory, args.output_directory, args.experiment_name)