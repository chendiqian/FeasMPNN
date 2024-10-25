# IPM-GNN2

# Environment setup

```angular2html
conda create -y -n ipmgnn python=3.11
conda activate ipmgnn
conda install -y pytorch==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.5.3  # maybe latest also works
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install wandb seaborn matplotlib cplex ortools hydra-core
conda install -y -c conda-forge qpsolvers 
```

# Reproduction of the results

## Main results 

python run.py --datapath DATA_TO_YOUR_INSTANCES --weight_decay 2.e-7 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --runs 3 --conv gcnconv

## Evaluation
During training, we fix the hparams for the log barrier function strength to tau = 0.01, tau_scale = 0.5, which generally works well, and we can find the best model. However, during the inference, a lot can be improved by playing around with these hyperparameters, especially when we want fewer evaluation steps. We list our option for different datasets.

| Dataset   | setc  | mis | cauc | fac   |
|-----------|-------| --- | --- |-------|
| tau       | 0.005 | 0.038 | 0.01 | 0.008 |
| tau scale | 0.85  | 0.63 | 0.85| 0.8   |