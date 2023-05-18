%%cu
#include <stdio.h>

#define M 500 // Number of rows of matrix A
#define N 500 // Number of columns of matrix B
#define K 500 // Number of columns of matrix A and rows of matrix B
#define TILE_WIDTH 32 // Tile width for sub-matrix multiplication

__global__ void matrixMultiply(int* A, int* B, int* C) {
    __shared__ int ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int sum = 0;
    for (int i = 0; i < (K-1)/TILE_WIDTH+1; i++) {
        if (row < M && i*TILE_WIDTH+tx < K) {
            ds_A[ty][tx] = A[row*K + i*TILE_WIDTH+tx];
        } else {
            ds_A[ty][tx] = 0;
        }

        if (col < N && i*TILE_WIDTH+ty < K) {
            ds_B[ty][tx] = B[(i*TILE_WIDTH+ty)*N + col];
        } else {
            ds_B[ty][tx] = 0;
        }

        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++) {
            sum += ds_A[ty][j] * ds_B[j][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row*N + col] = sum;
    }
}

int main() {
    int A[M * K];
    int B[K * N];
    int C[M * N] = {0};

    for (int i = 0; i < M * K; i++) {
        A[i] = i + 1;
    }

    for (int i = 0; i < K * N; i++) {
        B[i] = i + 1;
    }

    int* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(int));
    cudaMalloc((void**)&d_B, K * N * sizeof(int));
    cudaMalloc((void**)&d_C, M * N * sizeof(int));

    cudaMemcpy(d_A, A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N-1)/TILE_WIDTH+1, (M-1)/TILE_WIDTH+1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("Elapsed time: %f ms\n", milliseconds);

    return 0;
}
