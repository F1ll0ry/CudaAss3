%%cu
#include <stdio.h>

#define M 500 // Number of rows of matrix A
#define N 500 // Number of columns of matrix B
#define K 500 // Number of columns of matrix A and rows of matrix B

__global__ void matrixMultiply(int* A, int* B, int* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int sum = 0;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int A[M * K];
    int B[K * N];
    int C[M * N];

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

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

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
