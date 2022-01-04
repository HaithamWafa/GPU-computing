#include <stdlib.h>
#include <stdio.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <math.h>
#include <time.h>

__global__ void addMatKernel(int* d_matA, int* d_matB, int* d_matC, int width, int height) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	
	int k;

	if (i < height) {
		int offset = height * i;
		for (k = 0; k < height; k++)	d_matC[offset + k] = d_matA[offset + k] + d_matB[offset + k];
	}
}
int** generate_2D_array(int r, int c)
{
	int** row_ptr;
	int* ptr;
	ptr = (int*)malloc(r * c * sizeof(int));
	row_ptr = (int**)malloc(r * sizeof(int*));
	int i;
	for (i = 0; i < r; i++)
	{
		row_ptr[i] = ptr + (i * c);
	}

	return row_ptr;
}
void initializeArray2D(int** arr, int r, int c) {
	int i, j;

	for (i = 0; i < r; i++)
	{
		for (j = 0; j < c; j++) arr[i][j] = rand() % 512;
	}
}
void addMat(int *matA, int *matB, int *matC, int r, int c) {
	int* d_matA;
	int *d_matB;
	int *d_matC;
	int size = r * c * sizeof(int);
	cudaMalloc((void**)& d_matA, size);
	cudaMemcpy(d_matA, matA, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)& d_matB, size);
	cudaMemcpy(d_matB, matB, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)& d_matC, size);
	dim3 dimGrid(ceil(r / 16.0), 1, 1);
	dim3 dimBlock(16, 1, 1);
	addMatKernel << <dimGrid, dimBlock >> > (d_matA, d_matB,d_matC, r, c);
	cudaMemcpy(matC, d_matC, size, cudaMemcpyDeviceToHost);
	cudaFree(d_matA); cudaFree(d_matB); cudaFree(d_matC);

}
void printMatrix2D(int** mat, int rows, int cols) {
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			printf("%d    ", mat[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}
int main(void)
{
	int** matA;
	int** matB;
	int** matC;
	int r, c;
	time_t t;
	srand((unsigned)time(&t));
	r = rand()%10;  //ensuring size is randomized on 1-10 range for debugging
	while (r == 0 | r == 1) {
		r = rand() % 10;
	}
	c = r;
	
	matA = generate_2D_array(r, c);
	matB = generate_2D_array(r, c);
	matC=generate_2D_array(r,c);
	initializeArray2D(matA, r, c);
	initializeArray2D(matB, r, c);
	addMat(*matA, *matB, *matC, r, c);
	printf("random size = %d \n", r);
	printf("matrix A: \n");
	printMatrix2D(matA, r, c);
	printf("matrix B: \n");
	printMatrix2D(matB, r, c);
	printf("result: \n");
	printMatrix2D(matC, r, c);


	return 0;
}