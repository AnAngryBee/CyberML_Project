# CyberML_Project

Group Member:
* Yujun Kong (yk2272)
* Yuan Sha (ys3985)
* Tianshu Wang (tw2119)
* Xiaohan Chen (xc1598)

## Description

The interfaces of our repaired models are placed in corresponding eval_\*.ipynb

Our STRIP impelmentation enables us to predict images in a whole dataset or a single image with its involving dataset, since we need presence of other data to do superimposing for STRIP.

Our pruning + finetuning method works towards improving our backdoored network so that a backdoored input which originally belong to i will be less likely predicted as i. Please see more performance metrics about this in our report and code.

For the information about how to run our code, please refer to our comments within implementation.

## Environment

* tensorflow
* keras
* h5py
* tensorflow_model_optimization
