#ifndef CUDACC
#define CUDACC
#endif
#include <cuda.h>
#include <device_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CImg.h"

#define Nbins 5

using namespace cimg_library;


float kernelTime;

__global__ void Kernel_cuda(float* buffer, int size, float* histo)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int binSize = ceil(255.0 / Nbins);
		 __shared__ unsigned int histo_s[Nbins];
	if (threadIdx.x < Nbins) histo_s[threadIdx.x] = 0;
	__syncthreads();
	unsigned int prev_index = -1, curr_index;
	unsigned int accumulator = 0;

	for (unsigned int kk = tid; kk < size; kk += blockDim.x * gridDim.x)
	{
		int alphabet_position = (int)buffer[kk];

		if (alphabet_position >= 0 && alphabet_position <= 255) {

			curr_index = alphabet_position / binSize;
			//printf("%d,%d, %d ,%d\n",kk,curr_index,tid,alphabet_position);
			if (curr_index != prev_index)
			{
				if (accumulator > 0) atomicAdd(&(histo_s[prev_index]), accumulator);
				accumulator = 1;
				prev_index = curr_index;
			}
			else
			{
				accumulator++;
			}
		}
	}
	if (accumulator > 0) atomicAdd(&(histo_s[curr_index]), accumulator);
	__syncthreads();
	if (threadIdx.x < Nbins)
		atomicAdd(&(histo[threadIdx.x]), histo_s[threadIdx.x]);

}
void Histo_cuda(float* Image,int imgHeight, int imgWidth,float * hist)
{
	 float *d_out;
    float *d_hist;
 


	double s = clock();
	int size = imgHeight * imgWidth * sizeof(float);

	for (int i = 0; i < Nbins; i++)  hist[i] = 0;  //setting initial bins to zero

	 
	cudaError_t err = cudaMalloc((void**)&d_hist, Nbins * sizeof(float));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	 err = cudaMalloc((void**)&d_out, size );
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	err =  cudaMemcpy(d_out, Image, imgHeight * imgWidth * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err =  cudaMemcpy(d_hist, hist, Nbins * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
 

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	
	 Kernel_cuda <<< 1,Nbins >>>(d_out,imgWidth* imgHeight,d_hist);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&kernelTime, start, stop);


	//printf("Time taken executing kernel: %f \n", kernelTime);
	err =  cudaMemcpy(hist, d_hist, Nbins * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//double e = clock();
	//parallel_time = (double)(e - s) / CLOCKS_PER_SEC;



	 cudaFree(d_out);
    cudaFree(d_hist);

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
void SinitializeArray2D(float** arr, int r, int c) {  //ignore, used for testing
	int i, j;
	float fill[9] = { 255,255,255,255,1,10,40,23,140 };
	for (i = 0; i < r; i++)
	{
		for (j = 0; j < c; j++) arr[i][j] = fill[i*3+j];
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
			printf("%f    ", mat[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}


float* H_histogram(float** image, int img_rows, int img_cols)
{
	float sum; float* histo = (float*)malloc(Nbins * sizeof(float));
	int binSize = 256 / Nbins; int bin;
	int i, j;
	for (i = 0; i < Nbins; i++) histo[i] = 0;  //setting initial bins to zeros
	for (i = 0; i < img_rows; i++) {
		for (j = 0; j < img_cols; j++) {

			bin=image[i][j] / binSize;
			histo[bin] = histo[bin] + 1;
			
		}
	}
	return histo;
}

int main(void)
{
	float** H_image; float** H_outImg; float** D_image; float** D_outImg;
	//int Nbins;

	time_t t;
	srand((unsigned)time(&t));
	char imagePath[400];
	
 
	//float **test = generate_2D_array(3, 3); //used for testing
	//SinitializeArray2D(test, 3, 3);

	
	printf("please enter the image path\n");
	scanf("%s", imagePath);
	/*printf("please enter the number of bins you want in the histogram (will assume equal spacing)\n");
	scanf("%d", &Nbins);*/            //this variable bin size was disabled bec shared memory allocation in the kernel doesn't allow variable size

	float* CPUhistogram = (float*)malloc(Nbins * sizeof(float));
	float* GPUhistogram = (float*)malloc(Nbins * sizeof(float));


	printf("No of bins = %d\n", Nbins);
	//loading image and selecting one channel

	CImg< unsigned char> image((imagePath));
	int img_cols = image.width();
	int img_rows = image.height();
	image.channel(0);

	H_image = generate_2D_array(img_rows, img_cols);
	D_image = generate_2D_array(img_rows, img_cols);

	//copying image pixels to Host image and Device image input matrices
	int i, j;
	for (i = 0; i < img_rows; i++)
		for (j = 0; j < img_cols; j++) {
			H_image[i][j] = image(j, i);
			D_image[i][j] = image(j, i);
		}


	clock_t start = clock();  //timing host
	CPUhistogram = H_histogram(H_image, img_rows, img_cols);
	clock_t stop = clock();

	printf("the histogram of the input image is as follows(CPU): \n");
	for (i = 0; i < Nbins; i++) printf("%.f ", CPUhistogram[i]);

	double H_time = (double)(stop - start) / CLOCKS_PER_SEC;
	start = clock();  //timing wrapper function
	Histo_cuda(*D_image, img_rows, img_cols, GPUhistogram);
	stop = clock();
	double D_time = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("\nthe histogram of the input image is as follows(GPU): \n");
	for (int i = 0; i < Nbins; i++) printf("%.f ", GPUhistogram[i]);

	
	bool flag = 0;
	for (i = 0; i < Nbins; i++)
		if (CPUhistogram[i] != GPUhistogram[i])  //checking equivalency
				flag = 1;
			
	if (flag == 0) printf("\nDevice and Host produce similar results\n");

	printf("speedup/slow down achieved with memory overhead = time used by host (%.4f)/ time used by device (%.4f) = %.4f\n", H_time, D_time, H_time / D_time);
	printf("speedup/slow down achieved without memory overhead = time used by host (%.4f)/ time used by kernel (%.4f) = %.4f\n", H_time, kernelTime / 1000.0, H_time / (kernelTime / 1000.0));



	return 0;
}