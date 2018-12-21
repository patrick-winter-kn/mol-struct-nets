# mol-struct-nets
A framework to run experiments on convolutional neural networks that learn the classification of molecules based on a rasterized 2D representation.

## Dependencies
The code needs Python 3 and the following packages:

- numpy
- matplotlib
- progressbar2
- keras
- tensorflow-gpu

## Data
The following section will describe the necessary directory structure and the format the files have to be in.

### Directory Structure
The directory structure needs to look like this:

```
├── data_sets
|   ├── data_set_1
|   |   └── targets
|   |       ├── target_1
|   |       |   └── partitions
|   |       |       └── partition_1.h5
|   |       └── target_1.h5
|   └── data_set_1.h5
└── experiments
    ├── batch_1.csv
    └── experiment_1.msne
```

### Data Set Files
Data set files need to be in HDF5 Format. The file needs to contain a top level 1D string data set called `smiles` containing the SMILES string of each molecule.

### Target Files
Target files need to be in HDF5 Format. The file needs to contain a top level 2D integer data set called `classes`. The first dimension is the index of the molecule in the data set file while the second dimension is the class, with 0 for `active` and 1 for `inactive`. The value has to be 1 if the molecule is part of this class, 0 otherwise.

### Partition Files
Partition files need to be in HDF5 Format. The file needs to contain two top level 1D integer data sets called `test` and `train`. The `test` data set contains a list of indices (based on the molecules in the data set file) of the molecules contained in the test partition. `train` is the same for the train partition.

### Batch Files
A batch file is a CSV containing experiments that will be executed in succession. It starts with a set of parameters with the prefix `#`. With `# location <path>` the path of the experiment directory is specified. If the path is the location of the CSV file then this can be `# location same`. `# seeds <seeds>` specifies a set of random seeds. Each experiments will be run with each of these seeds. The seeds can either be a range of [x,y] written as `x-y`, or it can be a comma separated list. The rest of the file is one row for each experiment. The order of entries per line is the experiment file, the data set, the target and the partition. Not all of these need to be specified.

### Transfer Learning Data Sets Files
The transfer learning data sets file is a JSON file containing the list of data sets that are used for training and the list of data sets that are used for evaluation. An example looks like this:

```
{"train":[
["data_set_1","target_1"],
["data_set_1", "target_2"],
["data_set_2", "target_1"]
],
"eval":[
["data_set_3","target_1"],
["data_set_4","target_1"]
]}
```

## Scripts
The following section will explain the available scripts and how they are used. Most scripts also support the `--help` parameter to get info on available parameters.

### Creating and Editing Experiments
The create an experiment or edit an existing one the following command needs to be executed:

`python edit_experiment.py <experiment_path>`

`<experiment_path>` is the file path of the experiment. If omitted a selection dialog will be opened.

### Running Experiments
Single experiments are run using the following command:

`python run_experiment.py <experiment_path>`

`<experiment_path>` is the file path of the experiment. Look at the following table for additional parameters:

| Parameter | Description |
| --- | --- |
| `--data_set <data_set_name>` | Specify the name of the data set (excluding the `.h5`). |
| `--target <target_name>` | Specify the name of the target (excluding the `.h5`). |
| `--partition <partition_name>` | Specify the name of the partition (excluding the `.h5`). |
| `--seed <seed>` | Specify the random seed. |

### Running Experiment Batches
A batch of experiments can be run with the following command:

`python run_experiment_batch.py <batch_path>`

`<batch_path>` is the path to the batch CSV file containing the single experiments that should be executed. With the optional `--retries <number_retries>` parameter one can specifi the number of times an experiment should be retried if it fails.

### Running Transfer Learning Experiments
To run a transfer learning experiment, the following command is used:

`python run_transfer_experiment.py <experiment_path> <data_sets_path>`

`<experiment_path>` is the path to the experiment, while `<data_sets_path>` is the path to the transfer learning data sets file. The random seed can be set with the optional parameter `--seed <seed>`.
