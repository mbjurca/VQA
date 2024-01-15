# VQA
DeepLearning VUB 2023-2024 project 


The project has all parameters stored in the according .yaml files stored in the config directory.
If any hyperparameters (such as batch size, or learning rate) need to be changed simply modify the necessary
values in said files.


To run the project in order to train a new model and validate it on the validation subset use the command in the project's root


```bash
python ./scripts/train_vqa.py --exp_name name
```

Please make sure to change the argument passed as the experiment name appropriately.

To get the performances obtained for a random choice of answers run
```bash
python ./scripts/random_acc.py
```
