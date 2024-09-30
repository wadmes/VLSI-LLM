module load cudnn8.9-cuda12.3  cuda12.3/toolkit cuda12.3/fft cuda12.3/blas
conda activate molca
srun --pty --gres gpu:H100:1 /bin/bash
