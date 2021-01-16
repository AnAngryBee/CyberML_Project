# CyberML_Project

Group Members:
* Yujun Kong (yk2211)
* Yuan Sha (ys3985)
* Tianshu Wang (tw2119)
* Xiaohan Chen (xc1598)

## Description

The interfaces of our repaired models are placed in corresponding eval_\*.ipynb

Our STRIP impelmentation enables us to predict images in a whole dataset or a single image with its involving dataset, since we need presence of other data to do superimposing for STRIP.

Our pruning + finetuning method works towards improving our backdoored network so that a backdoored input which originally belong to i will be less likely predicted as i. Please see more performance metrics about this in our report and code.

In eval script, we only provide pruning method available for testing. Please refer to prune+strip_*.ipynb for more information about our STRIP and things.

## How to Run

To run our code, you need to firstly move corresponding bad nets in the root directory of our repo. Then run:

python3 <evaluation file> <image relative filename>

## Environment

* tensorflow
* keras
* h5py
* tensorflow_model_optimization
* numpy

P.S. No need to download data and models from our repo as we download them in our code directly from gdrive.
