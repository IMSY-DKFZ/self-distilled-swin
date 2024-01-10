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
conda create -n sdswin python=3.9
```
* Next, cd to the repo folder and install the requirements.txt file
```
pip install -r requirements.txt
```


# 2- Dataset
* **CholecT45:** You can request the dataset access [in the CholecT45 dataset's repository](https://github.com/CAMMA-public/cholect45).
* **Annotations:** The dataloader expects the annotations in a csv format, in order to generate the annotations csv file, run the following command

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




You'll need to adapt the `parent_path` and `output_dir` parameter in `config.yaml` with the path to the dataset in your local machine. For example:

```
parent_path: PATH/CholecT45
output_dir:  path where to save the outputs
```

# 3- Training
Once the environment and the path to the dataset are settled, the method is a 3 steps process: Train a teacher model, generate soft-labels, train the student model.

**NOTE: Make sure to use the parameter `exp` in each experiment to give a tag to your experiments. For ex: `exp=teacher`.**

### Step 1: Train a teacher model

```
python main.py target_size=131 epochs=20 distill=false exp=teacher
```
The checkpoints should be saved in the folder `output_dir/output/checkpoints` and the 5-Fold cross validation predictions in `output_dir/output/oofs`.

### Step 2: Generate the soft-labels
```
python softlabels.py target_size=131 exp=teacher
```
The soft-labels should be saved in the folder `parent_path/CholecT45/soflabels`
### Step 3: Train the student model
```
python main.py target_size=131 epochs=40 distill=true exp=student
```
The checkpoints should be saved in the folder `output_dir/output/checkpoints` and the 5-Fold cross validation predictions in `output_dir/output/oofs`.

# 4- Evaluation
**currently under development and will be made available shortly.**