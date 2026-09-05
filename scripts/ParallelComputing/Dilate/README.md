# CUDA Dilation Morphological Operator

Applies the [dilation morphological operator](https://en.wikipedia.org/wiki/Dilation_(morphology)) to a grayscale PGM image. For each pixel the operator replaces its value with the maximum gray level within a circular mask of radius `MASK_RADIUS = 5`. Two versions are provided: a serial CPU implementation (`dilate.c`) and a CUDA GPU implementation (`cuda-dilate.cu`).

## Files

| File | Description |
|---|---|
| `dilate.c` | Serial CPU implementation |
| `cuda-dilate.cu` | CUDA GPU implementation |
| `hpc.h` | Timing utilities |
| `dilate-cuda-job.sh` | SLURM batch script for the Leonardo HPC cluster |
| `cat-1344.pgm` | Sample input image (1344×1344 pixels) |

## Local compilation

### CPU version

Requires GCC with C99 support.

```bash
gcc -std=c99 -Wall -Wpedantic dilate.c -o dilate
```

### CUDA version

Requires NVIDIA CUDA Toolkit (`nvcc`).

```bash
nvcc -O2 cuda-dilate.cu -o cuda-dilate
```

## Running locally

Both programs read a PGM image from **stdin** and write the result to **stdout**.

```
./dilate [nrep] < input.pgm > output.pgm
./cuda-dilate [nrep] < input.pgm > output.pgm
```

- `nrep` — number of times the dilation operator is applied (default: 1)

**Example** — apply the operator 10 times to the sample image:

```bash
# CPU
./dilate 10 < cat-1344.pgm > cat-dilated.pgm

# GPU
./cuda-dilate 10 < cat-1344.pgm > cat-dilated-cuda.pgm
```

## Running on Leonardo (CINECA)

The script `dilate-cuda-job.sh` compiles and runs the CUDA version on an A100 GPU node.

```bash
# Upload sources to your home directory on Leonardo, then:
sbatch dilate-cuda-job.sh
```

SLURM resources requested by the script:

| Parameter | Value |
|---|---|
| Partition | `boost_usr_prod` |
| QOS | `boost_qos_dbg` |
| Account | `tra26_mhpcqc` |
| Nodes / tasks | 1 / 1 |
| GPUs | 1 × NVIDIA A100-SXM-64GB |
| Wall time | 10 min |

Output files are written to `cuda_dilate_<jobid>.out` and `cuda_dilate_<jobid>.err`.

The script runs a benchmark over several image sizes (`512`, `1024`, `2048`, `4096`, `8192` pixels square). Place the corresponding `random-input-<size>.pgm` files in the working directory to use them.

## Performance output format

The benchmark prints one CSV line per run:

```
width,height,rep,avg_time_s,Mpixels_s,GB_s
```

## Input image format

Input must be a binary PGM (`P5`) file — 8-bit grayscale. Any standard image editor or ImageMagick can convert to this format:

```bash
convert input.png -type Grayscale input.pgm
```
