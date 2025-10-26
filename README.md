# Complex-Weighted Convolutional Networks: Provable Expressiveness via Complex Diffusion
This repository contains the official code for the paper [**Complex-Weighted Convolutional Networks: Provable Expressiveness via Complex Diffusion**](https://openreview.net/forum?id=EgRvGMd2Zp#discussion) (The Fourth Learning on Graphs Conference 2025).
## Requirements

This project requires Python 3.12.9 and torch 2.5.1 and uses `conda` for environment management.

Create a new conda environment:
```bash
conda create -n name_env python=3.12.9
```
Activate environment:
```bash
conda activate name_env
````

Install torch 2.5.1 and CUDA 12.1:
```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Install requirements:
```bash
pip install networkx==3.4.2
pip install tqdm==4.67.1
pip install torch_sparse==0.6.18
pip install torch_geometric==2.6.1
pip install wandb==0.19.8
pip install torch_scatter==2.1.2
```
## Hyperparameters
To run a training, the following hyperparameters can be configured:

**dataset**: name of the dataset (texas/wisconsin/film/chameleon/cornell/citeseer/pubmed/cora)

**layers**: number of layers to apply diffusion

**hidden_channels**: number of complex hidden channels ($f$ in the paper)

**left_weights**=True if left weights ($\mathbf{W}_1^l$) are used

**right_weights**=True if left weights ($\mathbf{W}_2^l$) are used

**lr**: learning rate

**weight_decay**: weight decay for all the parameters except the ones ralated to the learning of complex weights

**complex_weight_decay**: weight decay for the parameters ralated to the learning of complex weights

**input_dropout**: dropout rate in the first linear layer

**dropout**: dropout rate during diffusion

**use_act**=True to enable the use of activation function

**complex_weights_act**: activation function used to learn complex weights ($\tilde\sigma$ in the paper)

**early_stopping**: number of patience epochs (stop training if validation does not improve after this number of consecutive epochs)

**epochs**: number of training epochs

**cuda**: ID of the GPU used (if any)

**folds**: number of folds

**batch_norm**=True if batch normalization is used

**entity**: wandb entity name

## Running examples
Training example:
```bash
python run.py --dataset=wisconsin --layers=4 --hidden_channels=8 --left_weights=True --right_weights=True --lr=0.02 --weight_decay=0.06 --input_dropout=0 --dropout=0.3 --use_act=True --complex_weights_act=tanh --early_stopping=200 --complex_weight_decay=0.26 --epochs=1000 --cuda=0 --batch_norm=True --entity=wandb-entity-name
```
wandb sweep example:
```bash
wandb sweep --project project-name sweep_example.yml
```
## Citation

For attribution in academic contexts, please cite the following paper:
```bibtex
@inproceedings{complexconv2025,
  title={Complex-Weighted Convolutional Networks: Provable Expressiveness via Complex Diffusion},
  author={Cristina López Amado, Tassilo Schwarz, Yu Tian, Renaud Lambiotte},
  booktitle={Proceedings of the Fourth Learning on Graphs Conference (LoG)},
  year={2025},
  url={https://openreview.net/forum?id=EgRvGMd2Zp}
}
