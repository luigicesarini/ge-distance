#!/bin/bash
#SBATCH --job-name=make_grid
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=256GB
#SBATCH --nodelist=cn08
#SBATCH --account=PR_climate

echo "start of job"
echo "eliminating results slurm older than 10 days"
date +'%Y-%m-%d %H:%M:%S'

DATE=$(date -d "11 days ago" +'%Y-%m-%d %H:%M:%S')

find /mnt/beegfs/lcesarini/ge-distance/src/distance/resultsSLURM/* -type f ! -newermt "$DATE" -exec rm {} +

export MPLCONFIGDIR=/mnt/beegfs/lcesarini/tmp/mat
WORKDIR=$PWD
cd $WORKDIR
echo $WORKDIR
module purge
conda init bash
source /home/luigi.cesarini/.bashrc
conda activate my_xclim_env

srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./make_grid.py --n_ul 100000 &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./make_grid.py --n_ul 10000 &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./make_grid.py --n_ul 1000 &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./make_grid.py --n_ul 10 &

wait
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m CNRM &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m HCLIMcom &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m ICTP &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p TRAIN -ref SPHERA -s SEQUENTIAL -a northern_italy -m KIT 

# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK julia ./graph/create_g.jl 
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p VALIDATION -ref SPHERA -s SEQUENTIAL -a northern_italy -m CMCC &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p VALIDATION -ref SPHERA -s SEQUENTIAL -a northern_italy -m CNRM &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p VALIDATION -ref SPHERA -s SEQUENTIAL -a northern_italy -m HCLIMcom &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p VALIDATION -ref SPHERA -s SEQUENTIAL -a northern_italy -m ICTP &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p VALIDATION -ref SPHERA -s SEQUENTIAL -a northern_italy -m KNMI &
# srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./bias/bias_corr_wind.py -p VALIDATION -ref SPHERA -s SEQUENTIAL -a northern_italy -m KIT 

date
echo "end of job"


