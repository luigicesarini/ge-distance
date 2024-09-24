#!/bin/bash
#SBATCH --job-name=make_grid
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --cpus-per-task=12
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=256GB
#SBATCH --nodelist=cn08
#SBATCH --account=PR_climate

echo "start of job"
echo "eliminating results slurm older than 10 days"
date +'%Y-%m-%d %H:%M:%S'

DATE=$(date -d "1 days ago" +'%Y-%m-%d %H:%M:%S')

find /mnt/beegfs/lcesarini/ge-distance/src/distance/resultsSLURM/* -type f ! -newermt "$DATE" -exec rm {} +

export MPLCONFIGDIR=/mnt/beegfs/lcesarini/tmp/mat
WORKDIR=$PWD
cd $WORKDIR
echo $WORKDIR
module purge
conda init bash
source /home/luigi.cesarini/.bashrc
conda activate my_xclim_env

srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./grid_csv_to_gpkg.py --n_ul 100000 &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./grid_csv_to_gpkg.py --n_ul 10000 &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./grid_csv_to_gpkg.py --n_ul 1000 &

wait

date
echo "end of job"


