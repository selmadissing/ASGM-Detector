#!/bin/bash
#SBATCH --job-name=asgm_train_no_osm_class_imbalance
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL,ALL
#SBATCH --mail-user=selma.dissing@student.uva.nl

# === Load only module-supported stack ===
module purge
module load 2023
module load GCC/12.3.0
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a
module load scikit-learn/1.3.1-gfbf-2023a

cd $HOME/mining_detector

# === Paths ===
INPUT_DIR=$HOME/mining_detector/256_no_osm
OUTPUT_DIR=$HOME/mining_detector/model_256_no_osm

# === Run training ===
python train_ensemble_class_imbalance.py --input_directory $INPUT_DIR --output_directory $OUTPUT_DIR --resolution 256