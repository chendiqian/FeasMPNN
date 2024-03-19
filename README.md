# IPM-GNN2

# Environment setup

```angular2html
conda create -y -n ipmgnn python=3.10
conda activate ipmgnn
conda install pytorch==2.0.0  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_scatter-2.1.1%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_sparse-0.6.17%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
pip install ml-collections
pip install wandb
```

# Reproduction of the results

## Main results 

python run.py --datapath DATA_TO_YOUR_INSTANCES --weight_decay 2.e-7 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --runs 3 --conv gcnconv

