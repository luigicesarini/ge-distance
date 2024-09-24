#!/bin/bash
#SBATCH --job-name=ensemble_test
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --cpus-per-task=20
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=256GB
#SBATCH --nodelist=cn08
#SBATCH --account=PR_climate

echo "start of job"
DAY=21
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

srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./create_ensemble.py --speed 'sparse' -nodes 10 -ce 4 -jn $SLURM_JOB_NAME -ng 95 &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./create_ensemble.py --speed 'sparse' -nodes 10 -ce 4 -jn $SLURM_JOB_NAME -ng 925 -pm &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./create_ensemble.py --speed 'sparse' -nodes 10 -ce 4 -jn $SLURM_JOB_NAME -ng 9174 -pm &

wait
date
echo "end of job"


