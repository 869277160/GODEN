<!--
 * @Author: your name
 * @Date: 2022-04-03 17:32:34
 * @LastEditTime: 2025-01-13 20:46:07
 * @LastEditors: wangding wangding19@mails.ucas.ac.cn
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /GODEN_Upload/Readme.md
-->

# GODEN

基于图神经常微分方程的信息传播预测

This repo provides a official implementation of **G**raph **N**eural **O**rdinary **D**ifferential **E**quation **N**etwork (**GODEN**) framework as described in the paper:

"""
Ding Wang, Wei Zhou, and Songiln Hu. 2024. Information Diffusion Prediction with Graph Neural Ordinary Differential Equation Network. In Proceedings of the 32nd ACM International Conference on Multimedia (MM '24). Association for Computing Machinery, New York, NY, USA, 9699–9708. https://doi.org/10.1145/3664647.3681363
"""

##  数据集 DATASET
We test our model on Four dataset, which are provided in the dataset folder,  they are from the [MS-HGAT](https://github.com/slingling/MS-HGAT). We pre-process the datasets (splitting and building graphs) before training, pre-process is shown in the main.py file and the preprocessed data is stored in the data folder.

## 环境配置 Environmental Settings

Our experiments are conducted on RedHat7, a single NVIDIA Tesla V100S GPU server. GODEN is implemented by `Python 3.10`, `Cuda 11.8`.

Create a virtual environment and install GPU-support packages via [Anaconda](https://www.anaconda.com/):
```shell
# create virtual environment
conda update -n base -c defaults conda
conda create --name py310_base python=3.10

pip install gpustat
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install torch torchvision torchaudio
pip install torch_geometric

# Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# If you have installed dgl-cudaXX.X package, please uninstall it first.
conda install -c dglteam/label/cu118 dgl

pip3 install einops

conda create -n GODEN --clone py310_base
```

or you can use pip to install the packages based on the requirements.txt file

```shell
pip install -r requirements.txt
```

## 运行命令 Usage

To train the model for the main experiment,  you can use the following commands:
```shell
conda activate GODEN
CUDA_VISIBLE_DEVICES=0 nohup python run.py --data="android" &
CUDA_VISIBLE_DEVICES=0 nohup python run.py --data="twitter" &
CUDA_VISIBLE_DEVICES=0 nohup python run.py --data="douban" &
CUDA_VISIBLE_DEVICES=0 nohup python run.py --data="memetracker" &
```

To train the model for some ablation experiment,  you can use the following commands:
```shell
conda activate GODEN
# ablation study on the Two main encoders 
DATASET=twitter
EPOCH=60
CUDA_VISIBLE_DEVICES=0 python main.py --data=${DATASET} --epoch=${EPOCH} --with_dy_graph=False --with_edge_recomputing=False --lab_no="ablation" --notes="Ablation_study: only_with_ST_GNN" 
CUDA_VISIBLE_DEVICES=0 python main.py --data=${DATASET} --epoch=${EPOCH} --with_st_graph=False --lab_no="ablation" --notes="Ablation_study: only_with_DY_ODE" 

# ablation study on the dy encoders 
CUDA_VISIBLE_DEVICES=0 python main.py --data=${DATASET} --epoch=${EPOCH} --with_normal_gnn=True --with_edge_recomputing=False --lab_no="ablation" --notes="Ablation_study: with_ST_GNN+_DY_GODE"  
CUDA_VISIBLE_DEVICES=0 python main.py --data=${DATASET} --epoch=${EPOCH} --with_normal_gnn=True --with_edge_recomputing=False --with_dy_normal_gnn=True --lab_no="ablation" --notes="Ablation_study: with_ST_GNN+_DY_GNN"  
CUDA_VISIBLE_DEVICES=0 python main.py --data=${DATASET} --epoch=${EPOCH} --with_normal_gnn=True --with_edge_recomputing=False --with_dy_normal_edge_gnn=True --lab_no="ablation" --notes="Ablation_study: with_ST_GNN+_DY_edge_GNN"  

# ablation on the dy graph merger
CUDA_VISIBLE_DEVICES=0 python main.py --data=${DATASET} --epoch=${EPOCH} --dy_merger=False --lab_no="ablation" --notes="Ablation_study: without_dy_merger"  

```

## Cite

If you find our paper & code are useful for your research, please consider citing us 😘:

```bibtex
@inproceedings{10.1145/3664647.3681363,
author = {Wang, Ding and Zhou, Wei and Hu, Songiln},
title = {Information Diffusion Prediction with Graph Neural Ordinary Differential Equation Network},
year = {2024},
isbn = {9798400706868},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3664647.3681363},
doi = {10.1145/3664647.3681363},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
pages = {9699–9708},
numpages = {10},
keywords = {graph neural network, information diffusion prediction, ordinary differential equations, social network},
location = {Melbourne VIC, Australia},
series = {MM '24}
}
```

## Contact

For any questions please open an issue or drop an email to: `wangding@iie.ac.cn`







