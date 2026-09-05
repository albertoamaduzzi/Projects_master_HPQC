/****************************************************************************
 *
 * cuda-dilate.cu - Dilation morphological operator (CUDA version)
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

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "hpc.h"

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Used by the PGM read/write routines */
    unsigned char *bmap; /* buffer of width*height bytes */
} PGM_image;

#define MASK_RADIUS 5
#define MASK_WIDTH (2*MASK_RADIUS+1)

#define BLOCK_SIZE 16

const unsigned char WHITE = 255;

/* Mask stored in constant memory: read-only, cached, fast broadcast */
__constant__ unsigned char d_mask[MASK_WIDTH * MASK_WIDTH];

/**
 * Initialize a PGM_image object.
 */
void init_pgm( PGM_image *img, int width, int height, unsigned char col )
{
    assert(img != NULL);
    img->width = width;
    img->height = height;
    img->maxgrey = 255;
    img->bmap = (unsigned char*)malloc(width * height);
    assert(img->bmap != NULL);
    memset(img->bmap, col, width * height);
}

/**
 * Read a PGM file from file `f`.
 */
void read_pgm( FILE *f, PGM_image *img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P5\n")) {
        fprintf(stderr, "Wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    sscanf(s, "%d %d", &(img->width), &(img->height));
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxgrey));
    if ( img->maxgrey > 255 ) {
        fprintf(stderr, "FATAL: maxgray=%d > 255\n", img->maxgrey);
        exit(EXIT_FAILURE);
    }
    img->bmap = (unsigned char*)malloc((img->width) * (img->height) * sizeof(unsigned char));
    assert(img->bmap != NULL);
    nread = fread(img->bmap, 1, (img->width) * (img->height), f);
    if ( (img->width) * (img->height) != nread ) {
        fprintf(stderr, "FATAL: error reading input: expecting %d bytes, got %d\n",
                (img->width) * (img->height), nread);
        exit(EXIT_FAILURE);
    }
}

/**
 * Write the image `img` to file `f`.
 */
void write_pgm( FILE *f, const PGM_image *img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);
    fprintf(f, "P5\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    fwrite(img->bmap, 1, (img->width) * (img->height), f);
}

/**
 * Free the bitmap of image `img`.
 */
void free_pgm( PGM_image *img )
{
    assert(img != NULL);
    free(img->bmap);
    img->bmap = NULL;
    img->width = img->height = img->maxgrey = -1;
}

/**
 * Initialize the mask (same as serial version: circle of radius 2).
 */
void init_mask( unsigned char *mask )
{
    for (int i = -MASK_RADIUS; i <= MASK_RADIUS; i++) {
        for (int j = -MASK_RADIUS; j <= MASK_RADIUS; j++) {
            mask[(MASK_RADIUS + i) * MASK_WIDTH + (MASK_RADIUS + j)] =
                (i * i + j * j <= 2 * 2);
        }
    }
}

/**
 * CUDA kernel: each thread computes the dilated value for one pixel (row, col).
 * The mask is read from constant memory.
 */
__global__ void dilate_kernel( const unsigned char *in, unsigned char *out,
                                int width, int height )
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width)
        return;

    int result = 255;
    for (int ry = -MASK_RADIUS; ry <= MASK_RADIUS; ry++) {
        for (int rx = -MASK_RADIUS; rx <= MASK_RADIUS; rx++) {
            const int nr = row + ry;
            const int nc = col + rx;
            if (nr >= 0 && nr < height && nc >= 0 && nc < width &&
                d_mask[(MASK_RADIUS + ry) * MASK_WIDTH + (MASK_RADIUS + rx)]) {
                const int val = in[nr * width + nc];
                if (val < result)
                    result = val;
            }
        }
    }
    out[row * width + col] = (unsigned char)result;
}

/**
 * Apply `rep` times the dilate operator using the GPU.
 */
void dilate( const PGM_image *in, PGM_image *out, const unsigned char *mask, int rep )
{
    const int width = in->width;
    const int height = in->height;
    const size_t size = (size_t)width * height;

    /* Copy mask to device constant memory */
    cudaSafeCall( cudaMemcpyToSymbol(d_mask, mask, MASK_WIDTH * MASK_WIDTH) );

    /* Allocate two device buffers for ping-pong iteration */
    unsigned char *d_buf[2];
    cudaSafeCall( cudaMalloc((void**)&d_buf[0], size) );
    cudaSafeCall( cudaMalloc((void**)&d_buf[1], size) );

    /* Copy input image to device buffer 0 */
    cudaSafeCall( cudaMemcpy(d_buf[0], in->bmap, size, cudaMemcpyHostToDevice) );

    /* Configure 2D grid and block */
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid((width  + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int cur = 0, next = 1;
    for (int r = 0; r < rep; r++) {
        dilate_kernel<<<grid, block>>>(d_buf[cur], d_buf[next], width, height);
        cudaCheckError();
        cur  = 1 - cur;
        next = 1 - cur;
    }

    /* Copy result back to host */
    cudaSafeCall( cudaMemcpy(out->bmap, d_buf[cur], size, cudaMemcpyDeviceToHost) );

    cudaSafeCall( cudaFree(d_buf[0]) );
    cudaSafeCall( cudaFree(d_buf[1]) );
}

/* Number of independent timing runs to average over (spec suggestion: ~5) */
#define NRUNS 5

int main( int argc, char *argv[] )
{
    PGM_image bmap, out;
    unsigned char mask[MASK_WIDTH * MASK_WIDTH];
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

    const int width  = bmap.width;
    const int height = bmap.height;

    /* --- Repeated timing runs (spec: repeat ~5 times to reduce noise) --- */
    double total_elapsed = 0.0;
    for (int run = 0; run < NRUNS; run++) {
        const double tstart = hpc_gettime();
        dilate(&bmap, &out, mask, rep);
        cudaSafeCall( cudaDeviceSynchronize() );
        total_elapsed += hpc_gettime() - tstart;
    }
    const double elapsed = total_elapsed / NRUNS;   /* average over runs */

    /* Throughput: total pixels processed per second across all rep iterations.
       Each iteration touches width*height pixels (read + write = 2 passes).
       We report:
         - Mpixels/s  : useful as an algorithmic throughput metric
         - GB/s       : useful as a memory bandwidth metric                  */
    const double total_pixels   = (double)width * height * rep;
    const double mpixels_per_s  = total_pixels / elapsed / 1.0e6;
    const double bandwidth_GBs  = 2.0 * total_pixels / elapsed / 1.0e9;  /* read + write */

    fprintf(stderr, "Image size      : %d x %d pixels\n", width, height);
    fprintf(stderr, "Iterations (rep): %d\n", rep);
    fprintf(stderr, "Timing runs     : %d\n", NRUNS);
    fprintf(stderr, "Avg exec time   : %.4f s\n", elapsed);
    fprintf(stderr, "Throughput      : %.2f Mpixels/s\n", mpixels_per_s);
    fprintf(stderr, "Bandwidth       : %.2f GB/s\n", bandwidth_GBs);
    /* CSV-friendly summary line for easy copy-paste into report tables */
    fprintf(stderr, "CSV: %d,%d,%d,%.4f,%.2f,%.2f\n",
            width, height, rep, elapsed, mpixels_per_s, bandwidth_GBs);

    write_pgm(stdout, &out, "produced by cuda-dilate");
    free_pgm(&bmap);
    free_pgm(&out);
    return EXIT_SUCCESS;
}
