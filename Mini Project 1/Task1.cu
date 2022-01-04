#include <stdlib.h>
#include <stdio.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <time.h>

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
	int k;
	int i = 4*( threadIdx.x + blockDim.x * blockIdx.x);
	for (k = 0; k < 4; k++)
	{
		if((i+k)<n) C[i+k] = A[i+k] + B[i+k];
	}

}

void vecAdd(float* A, float* B, float* C, int n) {
	int size = n * sizeof(float);
	float* d_A, * d_B, * d_C;
	cudaMalloc((void**)& d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)& d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)& d_C, size);
	vecAddKernel<<< ceil(n / 512.0), 512 >>> (d_A, d_B, d_C, n);
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
void initializeArray(float* arr, int N) {
	int i;
	for (i = 0; i < N; i++)
	{
		arr[i] = (float)rand();
	}
}
	void printVector(float* arr, int N, char name) {
		int i;
		printf("Vector %c: ", name);
		for (i = 0; i < N; i++)
		{
			printf("%.f  ", arr[i]);
		}
		printf("\n");
}



int main(void)
{
	int N;
	float* h_A;
	float* h_B;
	float* h_C;
	srand(time(0));
	N = rand()%10; //random size restricted to this range for debugging
	h_A = (float*)malloc(N * sizeof(float));
	h_B = (float*)malloc(N * sizeof(float));
	h_C = (float*)malloc(N * sizeof(float));


	int i;
	initializeArray(h_A,N); //randomly initializes the array to which the pointer is passed
	initializeArray(h_B, N);
	vecAdd(h_A, h_B, h_C, N);
	printf("Random size = %d\n", N);
	printVector(h_A, N, 'A');
	printVector(h_B, N, 'B');
	printVector(h_C, N, 'S');
   

	return 0;
}