#!/bin/bash

# Define the list of pretrained models
pretrained_models=('SwinT' 'SwinT+MultiT' 'SwinT+SelfDv2' 'SwinT+MultiT+SelfD' '+phase' 'SwinLarge')

echo "Self-distillation pretrained models inference started..."

# Iterate over the list and run the Python command for each element
for model in "${pretrained_models[@]}"; do
    echo "-----------------------------------------"
    echo "Inference using the pretrained model: $model"
    python generate.py inference=true pretrained_model=true save_folder=official_results exp="$model"
done

# Execute the evaluation command after the loop
echo "Starting evaluation with ensemble=true"
python evaluate.py save_folder=official_results ensemble=true ensemble_models="[+phase.csv,SwinLarge.csv,SwinT+MultiT+SelfD.csv]"
