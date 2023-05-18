%%writefile sequential.c
#include <stdio.h>
#include <time.h>

#define M 500 // Number of rows of matrix A
#define N 500 // Number of columns of matrix B
#define K 500 // Number of columns of matrix A and rows of matrix B

void matrixMultiply(int* A, int* B, int* C) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            int sum = 0;
            for (int i = 0; i < K; i++) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
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

    clock_t start = clock();

    matrixMultiply(A, B, C);

    clock_t end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    printf("Matrix multiplication completed.\n");
    printf("Elapsed time: %.2f ms\n", elapsed_time);

    return 0;
}
