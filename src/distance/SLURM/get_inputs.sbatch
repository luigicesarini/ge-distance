#!/bin/bash
#SBATCH --job-name=get_inputs
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1 
#SBATCH -n 3
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=256GB
#SBATCH --nodelist=cn08
#SBATCH --account=PR_climate

echo "start of job"
DAY=11
echo "eliminating results slurm older than $DAY days"
date +'%Y-%m-%d %H:%M:%S'

DATE=$(date -d "$DAY days ago" +'%Y-%m-%d %H:%M:%S')

find /mnt/beegfs/lcesarini/ge-distance/src/distance/resultsSLURM/* -type f ! -newermt "$DATE" -exec rm {} +

export MPLCONFIGDIR=/mnt/beegfs/lcesarini/tmp/mat
WORKDIR=$PWD
cd $WORKDIR
echo $WORKDIR
module purge
conda init bash
source /home/luigi.cesarini/.bashrc
conda activate my_xclim_env

srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./get_matrix_input.py --n_grid 95 &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./get_matrix_input.py --n_grid 925 &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./get_matrix_input.py --n_grid 9174 &

wait
date
echo "end of job"
