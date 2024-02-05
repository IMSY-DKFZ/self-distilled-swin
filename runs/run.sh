#!/bin/bash

# Initialize variables with default values
target_size=100  # Default value
exp=""
save_epoch=3    # Default value, change as needed
neplog=false     # Default value
model_name="swin_base_patch4_window7_224"  # Default value

# Parse command-line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    target_size=*)
      target_size="${1#*=}"
      ;;
    exp=*)
      exp="${1#*=}"
      ;;
    save_epoch=*)
      save_epoch="${1#*=}"
      ;;
    neplog=*)
      neplog="${1#*=}"
      ;;
    model_name=*)
      model_name="${1#*=}"
      ;;
    *)
      # Unknown argument
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
  shift
done

# Check if the required arguments are provided
if [ -z "$exp" ]; then
    echo "Usage: $0 exp=<value> [target_size=<value>] [save_epoch=<value>] [neplog=true|false] [model_name=swin_base_patch4_window7_224]"
    exit 1
fi

# Run the teacher model with save_epoch parameter, neplog flag, and model_name parameter
python main.py neplog="$neplog" model_name="$model_name" target_size="$target_size" epochs="$save_epoch" early_stopping=true distill=false exp="$exp" 

# Generate soft-labels using the teacher model
python generate.py inference=false target_size="$target_size" exp="$exp"

# Run the third Python command with save_epoch parameter and model_name parameter
python main.py neplog="$neplog" model_name="$model_name" target_size="$target_size" epochs=40 distill=true exp="${exp}+SelfD" 
