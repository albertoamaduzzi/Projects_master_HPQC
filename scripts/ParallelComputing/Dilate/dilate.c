/****************************************************************************
 *
 * dilate.c - Dilation morphological operator
 *
 * Copyright (C) 2026 Moreno Marzolla
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

/***
% Dilation morphological operator
% Claudio Gheller, [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-04-14

The [dilation morphological
operator](https://en.wikipedia.org/wiki/Dilation_(morphology)) is used
to expand the shapes contained in an image. The dilation operator is a
_stencil computation_ that can be implemented as follows. For each
pixel at coordinates $(i, j)$ we consider a square mask centered at
$(i, j)$ with radius `MASK_RADIUS` (in this implementation,
`MASK_RADIUS = 5`). Therefore, pixel $(i, j)$ is at the center of a
square of side `MASK_WIDTH = (2*MASK_RADIUS + 1)`.  We select the
pixels in the square where the mask is 1, and replace the gray level
of $(i, j)$ with the maximum gray level of the selected pixels. To
simulate synchronous update of all pixels, we make use of two bitmaps:
an input image and an output image. Pixels are _read_ from the input
bitmap, and _written_ into the output bitmap. Note that input pixels
near the border might have less that $\mathtt{MASK\_WIDTH} \times
\mathtt{MASK\_WIDTH}$ neighbors.

![](dilate.png){ width=80% }

This file provides a CPU implementation of the dilation operator. The
program reads an input image in
[PGM](https://en.wikipedia.org/wiki/Netpbm#PGM_example) (_Portable
Graymap_) format from standard input and writes the result to standard
output. The user can specify an optional parameter that denotes how
many times the operator should be applied (default 1). The mask is
circular and has a diameter of `MASK_RADIUS = 5` pixels.

The input format encodes grayscale images as a matrix of bytes, so
that each pixel can have a value from 0 (black) to 255
(white). Therefore, the input and output are rectangular matrices of
type `unsigned char`.

## Compiling on your machine

If you have a local Linux installation, you can compile as follows:

        gcc -std=c99 -Wall -Wpedantic dilate.c -o dilate

To execute:

        ./dilate [nrep] < input.pgm > output.pgm

where `nrep` is the number of times the dilate operator should be
applied, `input.pgm` and `output.pgm` are the input and output images,
respectively.

Example: to apply ten times the dilation operator:

        ./dilate 10 < cat-1344.pgm > cat-dilated.pgm

## Compiling on Leonardo

To compile the serial version for the CPU, create a script
`dilate-job.sh` with the following content:

```bash
#!/bin/bash
# dilate-job.sh

#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=tra26_mhpcqc
#SBATCH --partition=boost_usr_prod

module load gcc
gcc -std=c99 -Wall -Wpedantic dilate.c -o dilate
./dilate 10 < cat-1344.pgm > cat-dilated.pgm
```

and submit the job as usual:

```bash
sbatch dilate-job.sh
```

> **Note**: if you are using a large image or many iterations, you may
> need to increase the value of ``--time``, since the CPU version is
> quite slow.

To compile a CUDA version `cuda-dilate.cu`, create the following
script `dilate-cuda-job.sh`:

```bash
#!/bin/bash
# dilate-cuda-job.sh

#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=tra26_mhpcqc
#SBATCH --partition=boost_usr_prod

module load cuda
nvcc cuda-dilate.cu -o cuda-dilate
./cuda-dilate 10 < cat-1344.pgm > cat-dilated.pgm
```

and submit with `sbatch dilate-cuda-job.sh`.

# What to do

- Write **two** parallel implementations of this program, one that
  uses CUDA (call it `cuda-dilate.cu`), and one that uses OpenMP (call
  it `omp-dilate.c`); **optionally**, write also a version
  `openacc-dilate.c` that uses OpenACC; however, the OpenACC version
  is **not** required to pass the exam. The OpenMP version should
  involve a trivial modification of this program. The CUDA version is
  a bit more tricky; some suggestions are provided below.

- Analyze the performance of the CUDA version; to this aim, compute
  the execution time and throughput as a function of the image size
  and/or number of iterations. For simplicity, you can use square
  images only. A few random images of various sizes are provided; they
  have been generated using the following command on Linux:

  ```bash
  convert -size 4096x4096 -depth 8 plasma:fractal random-image-4986.pgm
  ```

  `convert` is part of the [ImageMagick](https://imagemagick.org/)
  package. Unfortunately, it seems that ImageMagick is not installed
  on Leonardo, so we provided some pre-generated input files for your
  convenience.

- Write a short report to briefly describe the parallel versions and
  the performance results.

> **Deadline**: 23:59 of may 15, 2026.


# How to hand in the assignment

To hand in the assignment, prepare .zip archive containing

- The source code of your programs, including all files required to
  compile the executable (e.g., the `hpc.h` file, if you use it);

- A text file called README containing compilation instructions;

- The report in PDF format.

Please **do not include any test image** since they are very big.
Send the archive to both instructors by email
(<moreno.marzolla@unibo.it>, <claudio.gheller@inaf.it>).


# Suggestions for the CUDA implementation

**Problem decomposition.** Given that the input is a 2D matrix of
numbers, we suggest to use a two-dimensional grid of two-dimensional
blocks. Each CUDA thread will apply the dilation operator to one pixel
of the input bitmap, and compute the result to be stored on the output
bitmap. There are no race conditions.

**Use of constant memory.** The mask is very small and does is not
modified by the CUDA threads. Therefore, it could be stored in
_constant memory_ on the device.

**Measuring the wall-clock time.** The GPU version will be quite
fast. To get a meaningful wall-clock time and throughput, we suggest
to use big images and apply the dilate operator many times using the
command-line parameter. Also, measurements should be repeated several
times (e.g., five) to level out statistical fluctuations.

# Files

- [dilate.c](dilate.c) [hpc.h](hpc.h)
- Input images:
  - [cat-1344.pgm](cat-1344.pgm)
  - [random-input-512.pgm](random-input-512.pgm)
  - [random-input-1024.pgm](random-input-1024.pgm)
  - [random-input-2048.pgm](random-input-2048.pgm)
  - [random-input-4096.pgm](random-input-4096.pgm)
  - [random-input-8192.pgm](random-input-8192.pgm)

***/

/* The following #define is required by hpc.h, since it uses the
   clock_gettime() function which is not part of ANSI C (but is a
   POSIX extension). */

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "hpc.h"
#include <string.h>

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Used by the PGM read/write routines */
    unsigned char *bmap; /* buffer of width*height bytes; each element represents the gray level of a pixel (0-255) */
} PGM_image;

#define MASK_RADIUS 5
#define MASK_WIDTH (2*MASK_RADIUS+1)

const unsigned char WHITE = 255;
/* const unsigned char BLACK = 0; */

/**
 * Initialize a PGM_image object: allocate space for a bitmap of size
 * `width` x `height`, and set all pixels to color `col`
 */
void init_pgm( PGM_image *img, int width, int height, unsigned char col )
{
    int i, j;

    assert(img != NULL);

    img->width = width;
    img->height = height;
    img->maxgrey = 255;
    img->bmap = (unsigned char*)malloc(width*height);
    assert(img->bmap != NULL);
    for (i=0; i<height; i++) {
        for (j=0; j<width; j++) {
            img->bmap[i*width + j] = col;
        }
    }
}

/**
 * Read a PGM file from file `f`. Warning: this function is not
 * robust: it may fail on legal PGM images, and may crash on invalid
 * files since no proper error checking is done.
 */
void read_pgm( FILE *f, PGM_image* img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P5") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P5\n")) {
        fprintf(stderr, "Wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxgrey; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxgrey));
    if ( img->maxgrey > 255 ) {
        fprintf(stderr, "FATAL: maxgray=%d > 255\n", img->maxgrey);
        exit(EXIT_FAILURE);
    }
    img->bmap = (unsigned char*)malloc((img->width)*(img->height)*sizeof(unsigned char));
    assert(img->bmap != NULL);
    /* Get the binary data from the file */
    nread = fread(img->bmap, 1, (img->width)*(img->height), f);
    if ( (img->width)*(img->height) != nread ) {
        fprintf(stderr, "FATAL: error reading input: expecting %d bytes, got %d\n", (img->width)*(img->height), nread);
        exit(EXIT_FAILURE);
    }
}

/**
 * Write the image `img` to file `f`; if not NULL, use the string
 * `comment` as metadata.
 */
void write_pgm( FILE *f, const PGM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P5\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    fwrite(img->bmap, 1, (img->width)*(img->height), f);
}

/**
 * Free the bitmap associated with image `img`; note that the
 * structure pointed to by `img` is NOT deallocated; only `img->bmap`
 * is.
 */
void free_pgm( PGM_image *img )
{
    assert(img != NULL);
    free(img->bmap);
    img->bmap = NULL; /* not necessary */
    img->width = img->height = img->maxgrey = -1;
}

int IDX(int row, int col, int width)
{
    return (row*width + col);
}

int imin(int a, int b)
{
    return (a < b ? a : b);
}


/**
 * Apply `rep` times the dilate operator to `in`; put the result in
 * `out`.
 */
void dilate( const PGM_image* in, PGM_image* out, const unsigned char *mask, int rep )
{
    const int width = in->width;
    const int height = in->height;
    const size_t size = width * height;

    /* We use two buffers to compute the iteration of the dilation
       operator. Initially, the input bitmap is copied to `buf[0]`.
       At each iteration, dilation is applied to `buf[cur]` and the
       result is stored into `buf[next]`, and then `cur` and `next`
       are exchanged. At the end, the result is copied into
       `out->bmap`. */

    unsigned char *buf[2];
    int cur = 0, next = 1;
    buf[0] = (unsigned char*)malloc(size); assert(buf[0] != NULL);
    buf[1] = (unsigned char*)malloc(size); assert(buf[1] != NULL);

    memcpy(buf[0], in->bmap, size);

    for (int r = 0; r < rep; r++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int result = 255;
                for (int ry = -MASK_RADIUS; ry <= MASK_RADIUS; ry++) {
                    for (int rx = -MASK_RADIUS; rx <= MASK_RADIUS; rx++) {
                        const int mask_row = MASK_RADIUS + ry;
                        const int mask_col = MASK_RADIUS + rx;
                        if ((row + ry) >= 0 && (row + ry < height) &&
                            (col + rx) >= 0 && (col + rx < width) &&
                            (mask[IDX(mask_row, mask_col, MASK_WIDTH)])) {
                            result = imin(result, buf[cur][IDX(row+ry, col+rx, width)]);
                        }
                    }
                }
                buf[next][IDX(row, col, width)] = result;
            }
        }
        /* exchange cur and next */
        cur = 1-cur;
        next = 1-cur;
    }
    /* Copy result to out->bmap. Note: the last result is in `buf[cur]`,
       not `buf[next]`. */
    memcpy(out->bmap, buf[cur], size);
    free(buf[0]);
    free(buf[1]);
}

/**
 * Initialize the mask. This function sets the mask as a circle of
 * radius 2, but any other initialization would be fine.
 */
void init_mask( unsigned char *mask )
{
    for (int i=-MASK_RADIUS; i<=MASK_RADIUS; i++) {
        for (int j = -MASK_RADIUS; j <= MASK_RADIUS; j++) {
            mask[ IDX(MASK_RADIUS+i, MASK_RADIUS+j, MASK_WIDTH) ] =
                (i*i + j*j <= 2*2);
        }
    }
}

int main( int argc, char* argv[] )
{
    PGM_image bmap, out;
    unsigned char mask[MASK_WIDTH*MASK_WIDTH];
    int rep = 1;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [rep] < in.pgm > out.pgm\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 )
        rep = atoi(argv[1]);

    init_mask(mask);
    read_pgm(stdin, &bmap);
    init_pgm(&out, bmap.width, bmap.height, WHITE);
    const double tstart = hpc_gettime();
    dilate(&bmap, &out, mask, rep);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "Execution time %.3f\n", elapsed);
    write_pgm(stdout, &out, "produced by dilate\n");
    free_pgm(&bmap);
    free_pgm(&out);
    return EXIT_SUCCESS;
}
