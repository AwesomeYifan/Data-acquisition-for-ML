# Data-acquisition-for-ML

The repository is being updated.

## Datasets:
CIFAR10 and CIFAR100 can be directly loaded by tensorflow.keras.datasets (see Utils.py). Crop and RoadNet can be found here: https://drive.google.com/drive/folders/1RyJs2yIqhKw_elxDgaeol3Mtg4inLH1c?usp=sharing.

Please put the downloaded datasets in folder `datasets` with their names unchanged.

## Folder Crop, roadnet, CIFAR10 and CIFAR100:
Each sub-folder correspondes to an initialization configuration (e.g., different values of u, different partitioning granularity). Examples for u=1 are given in each folder.

## To reproduce the results:

For Crop and RoadNet, directly run the corresponding test file. For CIFAR10 and CIFAR100, first run the test file, and a .csv file with the ids of the acquired images will be produced; then run the train_VGG.py file to train the network with the acquired images.
