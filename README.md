# DSA6000DProject

The final project for the DSAA6000D.\
Using 2 representitive algorithms to solve entity alignment problem\
MTransE: <https://arxiv.org/abs/1611.03954.pdf>\
GCN-Align: <https://aclanthology.org/D18-1032.pdf>

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

## Reference

This project was developed inspired by these following projects:

- [EA-for-KG](https://github.com/ruizhang-ai/EA_for_KG)
- [EAkit](https://github.com/THU-KEG/EAkit)
- [OpenEA](https://github.com/nju-websoft/OpenEA)

1. Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (n.d.). Translating Embeddings for Modeling Multi-relational Data.
2. Cao, Y., Liu, Z., Li, C., Liu, Z., Li, J., & Chua, T.-S. (2019). Multi-Channel Graph Neural Network for Entity Alignment. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 1452–1461. <https://doi.org/10.18653/v1/P19-1140>
3. Chen, M., Tian, Y., Yang, M., & Zaniolo, C. (2017). Multilingual Knowledge Graph Embeddings for Cross-lingual Knowledge Alignment (arXiv:1611.03954). arXiv. <http://arxiv.org/abs/1611.03954>
4. Ji, S., Pan, S., Cambria, E., Marttinen, P., & Yu, P. S. (2022). A Survey on Knowledge Graphs: Representation, Acquisition and Applications. IEEE Transactions on Neural Networks and Learning Systems, 33(2), 494–514. <https://doi.org/10.1109/TNNLS.2021.3070843>
5. Sun, Z., Hu, W., & Li, C. (2017). Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding (arXiv:1708.05045). arXiv. <http://arxiv.org/abs/1708.05045>
6. Wang, Z., Lv, Q., Lan, X., & Zhang, Y. (2018). Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 349–357. <https://doi.org/10.18653/v1/D18-1032>
7. Zhang, R., Trisedya, B. D., Li, M., Jiang, Y., & Qi, J. (2022). A benchmark and comprehensive survey on knowledge graph entity alignment via representation learning. The VLDB Journal, 31(5), 1143–1168. <https://doi.org/10.1007/s00778-022-00747-z>