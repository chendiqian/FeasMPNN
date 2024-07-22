# IPM-GNN2

# Environment setup

```angular2html
conda create -y -n ipmgnn python=3.11
conda activate ipmgnn
conda install -y pytorch==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.5.3  # maybe latest also works
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install wandb seaborn matplotlib
```

# Reproduction of the results

## Main results 

python run.py --datapath DATA_TO_YOUR_INSTANCES --weight_decay 2.e-7 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --runs 3 --conv gcnconv

