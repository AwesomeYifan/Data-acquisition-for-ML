# Data acquisition for improving Machine Learning models

The repository is being updated.

## Datasets:
CIFAR10 and CIFAR100 can be directly loaded by tensorflow.keras.datasets (see Utils.py). Crop and RoadNet can be found here: https://drive.google.com/drive/folders/1RyJs2yIqhKw_elxDgaeol3Mtg4inLH1c?usp=sharing.

Please put the downloaded datasets in folder `datasets` with their names unchanged.

## Folder Crop/roadnet/CIFAR10/CIFAR100:
Each sub-folder correspondes to an initialization configuration (e.g., different values of u, different partitioning granularity). Examples for u=1 are given in each folder. File init.csv contains the sample ids used to initialize the model (i.e., the consumer possesses before the data acquisition process).

## To reproduce the results:

For Crop and RoadNet, directly run the corresponding test file. As an example, if you run the code corresponding Crop with u=1, the final output (model accuracy) can be located at Crop/1/results. For CIFAR10 and CIFAR100, first run the test file, and a ABC.csv file with the ids of the acquired images will be produced; then run the train_VGG.py with the following command:

```
python3 train.py --dataset CIFAR10 --path_to_folder 'DIR_OF_THE_ABC.csv_FILE' --file_name 'ABC.csv'
```

 And the output can be located in 'DIR_OF_THE_ABC.csv_FILE/results/' (so be sure there is a results folder).
