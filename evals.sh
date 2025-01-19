#!/bin/bash
#SBATCH --account=log
#SBATCH --partition log_gpu_48gb
#SBATCH -N 1 # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00
#SBATCH -o "./slurm-output/slurm-%j.out" # where the output log will be stored
#SBATCH --mem=64000

cd IPM2

export pj=eval_port
export mpath=logs/ipm_rerun_port_bi_gin_exp0
export conv=gin
export use_tri=false

if $use_tri == true
then
  export gnn=tri
else
  export gnn=bi
fi


for steps in 16 32
do
    python evaluate_ipm.py datapath=/work/log1/chendi.qian/qp_port_800_0.01_ipm modelpath=$mpath wandb.project=$pj wandb.name=400_"$steps"step_"$conv"_"$gnn" wandb.enable=true ipm_eval_steps="$steps" conv="$conv"conv tripartite=$use_tri
    python evaluate_ipm.py datapath=/work/log1/chendi.qian/generalize/port/qp_test_port_1000_0.01_ipm modelpath=$mpath wandb.project=$pj wandb.name=600fixdense_"$steps"step_"$conv"_"$gnn" wandb.enable=true ipm_eval_steps="$steps" conv="$conv"conv tripartite=$use_tri
    python evaluate_ipm.py datapath=/work/log1/chendi.qian/generalize/port/qp_test_port_1200_0.01_ipm modelpath=$mpath wandb.project=$pj wandb.name=800fixdense_"$steps"step_"$conv"_"$gnn" wandb.enable=true ipm_eval_steps="$steps" conv="$conv"conv tripartite=$use_tri
    python evaluate_ipm.py datapath=/work/log1/chendi.qian/generalize/port/qp_test_port_1000_0.008_ipm modelpath=$mpath wandb.project=$pj wandb.name=600fixdeg_"$steps"step_"$conv"_"$gnn" wandb.enable=true ipm_eval_steps="$steps" conv="$conv"conv tripartite=$use_tri
    python evaluate_ipm.py datapath=/work/log1/chendi.qian/generalize/port/qp_test_port_1200_0.006_ipm modelpath=$mpath wandb.project=$pj wandb.name=800fixdeg_"$steps"step_"$conv"_"$gnn" wandb.enable=true ipm_eval_steps="$steps" conv="$conv"conv tripartite=$use_tri
done