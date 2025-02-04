# Towards graph neural networks for provably solving convex optimization problems

<img src="https://github.com/chendiqian/FeasMPNN/blob/qp/image.png" alt="drawing" width="900"/>
<p align="center">
An overview of our Feasibility-guaranteed iterations
</p>

## Environment setup

```angular2html
conda create -y -n ipmgnn python=3.11
conda activate ipmgnn
conda install -y pytorch==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.5.3  # maybe latest also works
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp311-cp311-linux_x86_64.whl
pip install wandb seaborn matplotlib cplex ortools hydra-core

# the next are only if you want to evaluate solvers
conda install -y -c conda-forge qpsolvers 
# for larger problems you need a license, please visit https://www.gurobi.com/ for more information
pip install gurobipy
```

## Dataset generation
For `generic`, `SVM` and `portfolio` QP problems, please see to `generate_instances.py` and `generate_instances.ipynb`. The instances are generated with the global node by default, but may or may not be used by different MPNNs. To run IPM methods, we need a trace of the interior points, please follow `qp2ipmgnn.ipynb`.

For large and sparse QP datasets for QPLIB pretraining, please see to `generate_foundation.ipynb`. To process QPLIB instances, please install [QPLIB](https://qplib.zib.de/doc.html) and process it with `QPLIB_prep.ipynb`.

For fixed size QP problems to compare with [IPM-LSTM](https://github.com/NetSysOpt/IPM-LSTM) and [DC3](https://github.com/locuslab/DC3), please see to `generate_homo_instance.ipynb`.

For LP instances, please see to `generate_instances_lp.py` and `generate_lp.ipynb`.

## MPNN

We provide code for heterogeneous [GCN](https://arxiv.org/abs/1609.02907), [GIN](https://arxiv.org/abs/1810.00826), [GATv2](https://arxiv.org/abs/2105.14491), [GCNII](https://arxiv.org/abs/2007.02133), [GEN](https://arxiv.org/abs/2006.07739) convolutions in `models/convs`.

By default, the datasets are always generated with a global node. You may specify the `tripartite` argument in the configs to indicate whether to use the global node or not. 

## Reproduction of the results
We provide command and configs in `configs` for reproducing results in the paper. Feel free to play with the config arguments. 

### Training
The pretrained model checkpoints are stored under `./logs` by default. 

Naive approach [Chen et al.](https://arxiv.org/abs/2209.12288)  
`python run_basegnn.py`

IPM-MPNN [Qian et al.](https://arxiv.org/abs/2310.10603), where the output of each layer corresponds to an interior point  
`python run_fixstep_ipmgnn.py`

For our IPM based method, where each iteration executes the whole MPNN and predicts a next point, the GPU memory cost is significantly larger, therefore we train on 4 cards of Nvidia L40S. The slurm configuration  
```angular2html
#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=4            # Number of tasks (GPUs) per node
#SBATCH --gpus-per-node=4              # GPUs per node
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00
#SBATCH -o "./slurm-output/slurm-%j.out" # where the output log will be stored
#SBATCH --mem=0

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  # Master node address
MASTER_PORT=12345                    # Master port
WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_PER_NODE))  # Total number of processes

# Run torchrun with the required arguments
torchrun --standalone --nproc_per_node=$SLURM_GPUS_PER_NODE --nnodes=$SLURM_NNODES \
         --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
          multigpu_ipm.py 
```

For our feasibility guaranteed method, just change the run file  
`torchrun --standalone --nproc_per_node=$SLURM_GPUS_PER_NODE --nnodes=$SLURM_NNODES \
         --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
          multigpu.py `

### Inference

After training, you can run the evaluation files to get the objective gap and violation statistics. Note that the dataset does not have to be the same as training data. Therefore, it enables generalization to larger instances or different data distributions.

Naive approach  
`python evaluate_basegnn.py modelpath=./logs/YOUR/PATH`

IPM-MPNN  
`python evaluate_fixstep_ipm.py`

Unlike training, our IPM based method can be executed on 1 single GPU  
`python evaluate_ipm.py`

Our Feasible method  
`python evaluate.py`

For QPLIB instances, we run the same file, process the individual QPLIB instances, and execute  
`python evaluate_qplib.py`. 

For benchmarking the inference time of solvers, please see to `evaluate_solver.py`, otherwise you don't have to install Gurobi license. 

### LP

LP is not the main selling point of the paper. Still, we provide code for LP training and inference with naive approach and our feasibility method. Please see to `run_basegnn_lp.py`, `run_lp.py` for training and `evaluate_lp.py` for inference. 
