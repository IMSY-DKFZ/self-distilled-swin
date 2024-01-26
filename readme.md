# Self-distillation for surgical action recognition

This repo serves as reproduction code for the paper "Self-Distillation for Surgical Action Recognition" see, [ArXiV](https://arxiv.org/abs/2303.12915) to be published at MICCAI 2023.

![](./figures/concept_overview.png)

<!-- <p align="center">
  <img src="./figures/concept_overview.png" alt="Figure">
</p> -->

The "Self-Distilled-Swin" project presents a Swin Transformer model trained through self-distillation. This approach is specifically designed to address challenges related to high number of classes, class imbalance and label ambiguity. The repository contains the implementation of this method applied to the CholecT45 dataset, as utilized in the CholecTriplet2022 challenge. The primary objective is to predict surgical action triplets: Instrument, Verb, and Target categories.

# 1- Environment set up

* First create a new environment:
```
conda create -n sdswin python=3.7.0
```
* Next, cd to the repo folder and install the requirements.txt file
```
pip install -r requirements.txt
```


# 2- Dataset
### **CholecT45:** 
You can request the dataset access [in the CholecT45 dataset's repository](https://github.com/CAMMA-public/cholect45).


After downloading the dataset, you'll need to adapt the `parent_path` and `output_dir` parameter in `config.yaml` with the path to the dataset in your local machine. For example:

```
parent_path: PATH/CholecT45
output_dir:  path where to save the outputs
```

### **Annotations:** 
The dataloader expects the annotations in a csv format, in order to generate the annotations csv file, run the following command

```
python parse.py
```


Once the CholecT45.csv file is generated, the final CholecT45 folder structure should be as following:
- CholecT45
  - data
    - VID01
    - VID02
    - ...
  - dict
  - instrument
  - target
  - triplet
  - verb
  - dataframes
    - CholecT45.csv




# 3- Training
Once the environment and the path to the dataset are settled, the method is a 3 steps process: Train a teacher model, generate soft-labels, train the student model.

* **NOTE: Make sure to use the parameter `exp` in each experiment to give a tag to your experiments. For ex: `exp=teacher`.**


### Step 1: Train a teacher model

```
python main.py target_size=131 epochs=20 distill=false exp=teacher
```
The checkpoints will be saved in the folder `output_dir/checkpoints` and the 5-Fold cross validation predictions in `output_dir/oofs`.

### Step 2: Generate the soft-labels
Make sure to use use same tags used when training the teacher model mainly `target_size` and `model_name`.
```
python generate.py inference=false target_size=131 exp=teacher
```
The soft-labels will be saved in the folder `parent_path/CholecT45/soflabels`
### Step 3: Train the student model
```
python main.py target_size=131 epochs=40 distill=true exp=student
```
The checkpoints will be saved in the folder `output_dir/checkpoints` and the 5-Fold cross validation predictions in `output_dir/oofs`.

# 4- Reproduce the paper experiments
The paper experiments reproduction guide:

### SwinT: 
SwinT is a baseline Swin base transformer trained with only 100 triplets. To train using 100 triplets, set `target_size=100`.
```
python main.py target_size=100 epochs=20 distill=false exp=SwinT
```
### SwinT+Multi: 
SwinT+Multi is a Swin base transformer baseline trained in a multitask learning fashion using additional instrument, verb, and target annotations. To use the additional annotations, set `target_size=131`.
```
python main.py target_size=131 epochs=20 distill=false exp=SwinT+MultiT
```
### SwinT+SelfD: 
SwinT+SelfD is a Swin base transformer trained with self-distillation using soft-labels. Generate soft-labels first using:
```
python generate.py inference=false target_size=100 exp=SwinT
```
Then train the model using `distill=true`

```
python main.py target_size=100 epochs=40 distill=true exp=SwinT+SelfD
```
### SwinT+MultiT+SelfD: 
SwinT+MultiT+SelfD is a Swin base transformer trained using multitask learning and self-distillation. Generate soft-labels for the SwinT+MultiT model:
```
python generate.py inference=false target_size=131 exp=SwinT+MultiT
```
Then, train the student model using `distill=true`
```
python main.py target_size=131 epochs=40 distill=true exp=SwinT+MultiT+SelfD
```
### Ensemble:
Ensemble model information will be made available shortly.


# 5- Inference
To use saved checkpoints for inference, set `inference=true`
```
python generate.py inference=true target_size=100 exp=SwinT
```

Ensure all five folds' checkpoints are available at `output_dir/checkpoints`. The predictions will be saved in `output_dir/predictions` folder.

**Our model checkpoints will be made available shortly for public use**

# 6- Evaluation
Final predictions are saved either in the `output_dir/oofs` folder after training or the `output_dir/predictions` folder if generated during inference. To evaluate training predictions, set `inference=false`:



```
python evaluate.py inference=false
```
To evaluate inference predictions, set `inference=true`
```
python evaluate.py inference=true
```
All saved experiments in the respective folder will be evaluated.

If testing the ensemble of multiple experiments, set `ensemble=true` add the relative path to those predictions in a list `ensemble_models=[ swin_bas_131_SwinT+MultiT+SelfD.csv, swin_bas_100_SwinT.csv]`