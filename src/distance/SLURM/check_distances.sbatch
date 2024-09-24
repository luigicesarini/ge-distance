#!/bin/bash
#SBATCH --job-name=check_distances
#SBATCH -e "resultsSLURM/%x-%j.err"
#SBATCH -o "resultsSLURM/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=256GB
#SBATCH --nodelist=cn08
#SBATCH --account=PR_climate

echo "start of job"
DAY=2
echo "eliminating results slurm older than $DAY days"clear
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

# N_GRID=11617
N_GRID=925

r_index_grid=($(shuf -i 0-$N_GRID -n 48 | sort -n))

N_ITER=10

srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[0]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[1]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[2]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[3]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[4]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[5]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[6]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[7]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[8]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[9]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[10]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[11]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[12]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[13]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[14]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[15]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[16]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[17]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[18]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[19]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[20]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[21]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[22]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[23]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[24]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[25]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[26]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[27]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[28]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[29]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[30]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[31]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[32]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[33]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[34]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[35]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[36]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[37]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[38]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[39]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[40]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[41]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[42]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[43]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[44]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[45]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[46]} --n_grid $N_GRID --n_iter $N_ITER &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python ./control_distances.py --index_grid ${r_index_grid[47]} --n_grid $N_GRID --n_iter $N_ITER &


wait

date
echo "end of job"


