# DSA6000DProject

The final project for the DSAA6000D.\
Using 2 representitive algorithms to solve entity alignment problem\
GitHub URL: <https://github.com/BellaZ98/6000DProject>

## Install Requirements

```bash
pip install -r requirements.txt
```

## Experiment

After setting up the environment, you can open the run.py and run.

### Set up the Tensorboard

Open the terminal in the project folder,

```bash
tensorboard --logdir='./_runs/'
```

If you want to check the output of the experiments result before, change the logdir into `./_runs/initial_bk`, modify the command line as follows:

```bash
tensorboard --logdir='./_runs/initial_bk'
```

If you want to modify the configuration of the model, just change the json files in `./training/configuration` and change the file_root in `run.py` if needed.

## Tips for configuration

If you want to change the model or the configuration to run, just change the `config_file` in `main()` of `run.py`.
If you want to modify the training process, here are the meanings of some lines of configuration

```json
{
  "log" : "mtranse",
  "data" : "DBP15K",
  "data_dir" : "data/DBP15K/zh_en",
  "rate" : 0.3,
  "epoch" : 1000,
  "check" : 10,
  "update" : 10,
  "train_batch_size" : 20000,
  "encoder" : "",
  "hiddens" : "100",
  "decoder" : "TransE,MTransE_Align",
  "sampling" : ".,.",
  "k" : "0,0",
  "margin" : "0,0",
  "alpha" : "1,50",
  "feat_drop" : 0.0,
  "lr" : 0.01,
  "seed" : 6000,
  "train_dist" : "euclidean",
  "test_dist" : "euclidean"
}
```

The line that should to be modified during experiments:\
`log` : model name and the log name,\
`data_dir` : the route to the dataset,\
`rate` : the seed rate of training set, 0.3 means using 30% for training and 70% for test\
`epoch` : number of epochs,\
`seed` : random seed of spliting the training set and test set.
