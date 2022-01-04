#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include<iostream>
#include<cmath>
#include<time.h>

#define VEC_SIZE 67108864
#define BLOCK_WIDTH 1024
float kernelTime;

__global__ void SumReduceKernel(float* X) {

	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride;

	unsigned int t = threadIdx.x;
	__shared__ float partialSum[BLOCK_WIDTH];
	partialSum[t] = X[index];
	
	for (stride = 1; stride < blockDim.x; stride *= 2)
	{
		__syncthreads();
		if (t % (2 * stride) == 0)
			partialSum[t] += partialSum[t + stride];
	}

	X[blockIdx.x] = partialSum[0];

}

float D_SumReduce(float* h_X)
{
	float sum;
	float* d_X;

	cudaError_t err = cudaMalloc(&d_X, VEC_SIZE);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_X, h_X, VEC_SIZE, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	dim3 dimBlock(BLOCK_WIDTH, 1, 1);
	dim3 dimGrid(ceil(float(VEC_SIZE) / BLOCK_WIDTH), 1, 1);
	//printf("that is: %d", blockDim.x);
	int N = VEC_SIZE;
	while (N > 1) {
		SumReduceKernel << < dimBlock, BLOCK_WIDTH >> > (d_X);
		N = ceil(N / 1024.0);
	}

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&kernelTime, start, stop);
	printf("Kernel time is =  %f ms\n", kernelTime);

	err = cudaMemcpy(&sum, d_X, sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	return sum;

}


void initializeArray(float* arr) {
	int i;

	for (i = 0; i < VEC_SIZE; i++)
	{
		 arr[i] = rand() % 10;
	}
}

void ZeroInitArray(float* arr) {
	int i;

	for (i = 0; i < VEC_SIZE; i++)
	{
		arr[i] = 0;
	}
}
void printVector(float* X) {
	int i;
	for (i = 0; i < VEC_SIZE; i++)  printf("%f", X[i]);
	printf("\n");
}

float H_SumReduce(float*X) {
	float sum=0; int i;
	for (i=0; i < VEC_SIZE; ++i) sum = sum + X[i];
	return sum;
}

int main(void)
{
	
	float* X = (float*)malloc(VEC_SIZE * sizeof(float));
	float H_sum;
	float D_sum;

	time_t t;
	srand((unsigned)time(&t));
	
	initializeArray(X);

	printf("Vector size is %d \n", VEC_SIZE);
	//printVector(X);
	clock_t start = clock();  //timing host
	H_sum = H_SumReduce(X);
	clock_t stop = clock();
	double H_time = (double)(stop - start) / CLOCKS_PER_SEC;
	start = clock();  //timing wrapper function
	D_sum=D_SumReduce(X);
	stop = clock();
	double D_time = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("speedup/slow down achieved with memory overhead = time used by host (%.5f)/ time used by device (%.5f) = %.5f\n", H_time, D_time, H_time / D_time);
	printf("speedup/slow down achieved without memory overhead = time used by host (%.5f)/ time used by kernel (%.5f) = %.5f\n", H_time, kernelTime / 1000.0, H_time / (kernelTime / 1000.0));
	if (D_sum = H_sum) 
		printf("CPU and GPU produce similar sums (verified).");
	else
		printf("CPU and GPU produce different sums! Re-check.");

	

	return 0;
}