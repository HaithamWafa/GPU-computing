#include <stdlib.h>
#include <stdio.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <math.h>
#include <time.h>

__global__ void pixelMultiplierKernel(int *Pin, int *Pout, int width, int height) {
	
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	if (col < width && row < height) {
		int offset = row * width + col;
		Pout[offset] = Pin[offset] * 3;
	}

}

void pixelMult(int *h_Pin, int *h_Pout, int r, int c) {
	int size = r*c* sizeof(int);
	int* Pin; 
	int* Pout;
	cudaMalloc((void**)& Pin, size);
	cudaMemcpy(Pin, h_Pin, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)& Pout, size);
	dim3 dimGrid(ceil(r / 16.0), ceil(c / 16.0), 1);
	dim3 dimBlock(16, 16, 1);
	pixelMultiplierKernel << <dimGrid, dimBlock >> > (Pin, Pout, r, c);
	cudaMemcpy(h_Pout, Pout, size, cudaMemcpyDeviceToHost);
	cudaFree(Pin); cudaFree(Pout); 
}
void initializeArray2D(int** arr, int r, int c) {
	int i, j;
	time_t t;
	srand((unsigned)time(&t));
	for (i = 0; i < r; i++)
	{
		for (j = 0; j < c; j++) arr[i][j] = rand() % 256;
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
void printMatrix2D(int** mat, int rows, int cols) {
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			printf("%d   ", mat[i][j]);
		}
		printf("\n");
	}
}


int main(void)
{
	int** image;
	int r, c;
	int** h_Pout;

	//1000x800 pixel image  
	r = 1000;
	c = 800;

	image = generate_2D_array(r, c);   //input image construction
	initializeArray2D(image, r, c);  //random initilization
	h_Pout=generate_2D_array( r, c); //host side output image

	pixelMult(*image, *h_Pout, r, c); 
	printf("Initial image: \n");
	printMatrix2D(image, r, c);
	printf("after running kernel: \n");
	printMatrix2D(h_Pout, r, c);


	return 0;
}