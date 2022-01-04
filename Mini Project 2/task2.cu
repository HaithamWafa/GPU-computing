#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define R1 2000
#define C1 1500
#define C2 2300
#define BLOCK_WIDTH 32
float kernelTime;

__global__ void MatrixMulKernel(float* A, float* B, float* P, int M, int N, int L) {

	float Pvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int k;
	if (row < M && col < L) {
		for (k = 0; k < N; ++k)

		{

			int A_element = A[row * N + k];

			int B_element = B[k * L + col];

			Pvalue += A_element * B_element;


		}

		P[row * L + col] = Pvalue;
	}

}
void D_multiplyMat(float* h_matA, float* h_matB, float* h_matC, int M, int N, int L)
{
	int sizeA = M * N * sizeof(float);
	int sizeB = N * L * sizeof(float);
	int sizeC = M * L * sizeof(float);
	float* matA;
	float* matB;
	float* matC;
	cudaError_t err = cudaMalloc((void**)& matA, sizeA);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(matA, h_matA, sizeA, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void**)& matB, sizeB);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(matB, h_matB, sizeB, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void**)& matC, sizeC);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 dimGrid(ceil((C2 + BLOCK_WIDTH - 1) / (float)BLOCK_WIDTH), ceil((R1 + BLOCK_WIDTH - 1) / (float)BLOCK_WIDTH),1);
	//printf("that is: %d", blockDim.x);
	
	MatrixMulKernel <<< dimGrid, dimBlock >> > (matA, matB, matC, M, N, L);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&kernelTime, start, stop);
	printf("Kernel time is =  %f ms\n", kernelTime);
	cudaMemcpy(h_matC, matC, sizeC, cudaMemcpyDeviceToHost);
	cudaFree(matA); cudaFree(matC); cudaFree(matC);

}

float** generate_2D_array(int r, int c)
{
	float** row_ptr;
	float* ptr;
	ptr = (float*)malloc(r * c * sizeof(float));
	row_ptr = (float**)malloc(r * sizeof(float*));
	int i;
	for (i = 0; i < r; i++)
	{
		row_ptr[i] = ptr + (i * c);
	}

	return row_ptr;
}
void initializeArray2D(float** arr, int r, int c) {
	int i, j;

	for (i = 0; i < r; i++)
	{
		for (j = 0; j < c; j++) arr[i][j] = rand() % 10;
	}
}

void ZeroInitArray2D(float** arr, int r, int c) {
	int i, j;

	for (i = 0; i < r; i++)
	{
		for (j = 0; j < c; j++) arr[i][j] = 0;
	}
}
void printMatrix2D(float** mat, int rows, int cols) {
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
float** H_multiplyMat(float** a, float** b, float** c) {
	int i, j, k;
	for (i = 0; i < R1; ++i)
		for (j = 0; j < C2; ++j)
			for (k = 0; k < C1; ++k)
			{
				c[i][j] += a[i][k] * b[k][j];
			}

	return c;


}

int main(void)
{
	float** matA;
	float** matB;
	float** matC;

	time_t t;
	srand((unsigned)time(&t));
	matA = generate_2D_array(R1, C1);
	matB = generate_2D_array(C1, C2);
	matC = generate_2D_array(R1, C2);
	initializeArray2D(matA, R1, C1);
	initializeArray2D(matB, C1, C2);
	ZeroInitArray2D(matC, R1, C2);

	/*printf("matrix A: \n");
	/printMatrix2D(matA, R1, C1);
	/printf("matrix B: \n");
	/printMatrix2D(matB, C1, C2);*/
	clock_t start = clock();  //timing host
	matC = H_multiplyMat(matA, matB, matC);
	clock_t stop = clock();
	double H_time = (double)(stop - start) / CLOCKS_PER_SEC;
	 start = clock();  //timing wrapper function
	D_multiplyMat(*matA, *matB, *matC, R1, C1, C2);
	 stop = clock();
	double D_time = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("speedup/slow down achieved with memory overhead = time used by host (%.4f)/ time used by device (%.4f) = %.4f\n", H_time , D_time,  H_time/D_time);
	printf("speedup/slow down achieved without memory overhead = time used by host (%.4f)/ time used by kernel (%.4f) = %.4f\n",H_time, kernelTime / 1000.0, H_time/(kernelTime / 1000.0));
	double GFOP = 2*(R1 * C2 / float(pow(10, 9))) * C1;
	double H_GFLOPS = GFOP / H_time;
	double D_GFLOPS = GFOP / D_time;
	printf("the host was capable of running %.4f GFLOPS, while the device was capable of %.4f GFLOPS\n", H_GFLOPS, D_GFLOPS);
	float GPU_util = ((BLOCK_WIDTH * BLOCK_WIDTH)/2048.0)*100;
	printf("GPU utilization as calculated by number of active threads/SM = %.4f %%\n", GPU_util);
	//printf("result: \n");
	//printMatrix2D(matC, R1, C2);
	
	return 0;
} 