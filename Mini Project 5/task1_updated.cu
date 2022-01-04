#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include<cuda.h>
#include<cmath>
#include<time.h>
#include "CImg.h"
using namespace cimg_library;


float kernelTime;

//int  img_cols = 4;
//int  img_rows = 4;


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


__global__ void col_kernel(float* input, float* out, int size, int g, int w)
{
	extern __shared__ int T[];
	int temp[300];
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int i = bx + tx * w;
	int index = g * i;

	for (int k = 0; k < g; k++)
		if (((k + (tx * g)) < size) && index + k < size * w)
			T[tx * g + k] = input[index + k];

	__syncthreads();

	for (int stride = 1; stride < blockDim.x * g; stride *= 2) {
		for (int k = 0; k < g; k++)
			if ((((tx * g) + k) >= stride) && ((tx * g) + k) < size)
				temp[k] = T[tx * g + k] + T[tx * g + k - stride];
		__syncthreads();
		for (int k = 0; k < g; k++)
			if ((((tx * g) + k) >= stride) && ((tx * g) + k) < size)
				T[tx * g + k] = temp[k];
		__syncthreads();
	}
	for (int k = 0; k < g; k++)
		if (tx * g + k < size && ((index + k) < (size * w)))
			out[index + k] = T[tx * g + k];
	__syncthreads();
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
			printf("%f    ", mat[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

__global__ void row_kernel(float* input, float* out, int size, int g, int H)
{

	extern __shared__ int T[];
	int temp[300];
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int i = bx * blockDim.x + tx;
	int index = g * i;

	for (int k = 0; k < g; k++)
		if (((k + (tx * g)) < size) && index + k < size * H)
			T[tx * g + k] = input[index + k];

	__syncthreads();

	for (int stride = 1; stride < blockDim.x * g; stride *= 2) {
		for (int k = 0; k < g; k++)
			if ((((tx * g) + k) >= stride) && ((tx * g) + k) < size)
				temp[k] = T[tx * g + k] + T[tx * g + k - stride];
		__syncthreads();
		for (int k = 0; k < g; k++)
			if ((((tx * g) + k) >= stride) && ((tx * g) + k) < size)
				T[tx * g + k] = temp[k];
		__syncthreads();
	}
	for (int k = 0; k < g; k++)
		if (tx * g + k < size && ((index + k) < (size * H)))
			out[index + k] = T[tx * g + k];
	__syncthreads();
}


float D_SATfunc(float* image, float* SAT, int x1, int y1, int x2, int y2, int n, int m)
{

	int size = n * m;
	int granularity = ceil(m / 1024.0);
	int granularity2 = ceil(n / 1024.0);
	int size2 = m * sizeof(float);
	int G1 = (ceil(m / (double)granularity));
	int G2 = (ceil(n / (double)granularity2));

	float* D_image, * D_SAT;

	cudaError_t err = cudaMalloc((void**)& D_image, sizeof(float) * size);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc(&D_SAT, sizeof(float) * size);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(D_image, image, size * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	dim3 grid(n, 1, 1);
	dim3 blocks(G1, 1, 1);
	dim3 grid2(m, 1, 1);
	dim3 blocks1(G2, 1, 1);
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	row_kernel << <grid, blocks, size2 >> > (D_image, D_SAT, m, granularity, n);
	cudaDeviceSynchronize();
	cudaEventCreate(&end);
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&kernelTime, start, end);


	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	col_kernel << <grid2, blocks1, size2 >> > (D_SAT, D_image, n, granularity2, m);
	cudaDeviceSynchronize();
	cudaEventCreate(&end);
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&kernelTime, start, end);

	cudaThreadSynchronize();

	err = cudaMemcpy(SAT, D_image, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	int indexA = x1 * m + y1;
	int indexD = x2 * m + y2;
	int indexB = x2 * m + y1;
	int indexC = x1 * m + y2;


	float sum = SAT[indexA] + SAT[indexD] - SAT[indexB] - SAT[indexC];

	cudaFree(D_image);
	cudaFree(D_SAT);



	return sum;
}
float H_SATfunc(float* image, int x1, int y1, int x2, int y2, int n, int m)
{

	float* H_SAT = new float[n * m];
	int x, y, z;



	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++) {

			if (((i - 1) < 0) && (j - 1) < 0) {
				x = 0;
				y = 0;
				z = 0;
			}
			else
				if ((i - 1) < 0) {
					x = 0;
					y = H_SAT[i * m + (j - 1)];
					z = 0;

				}
				else
					if ((j - 1) < 0) {
						x = H_SAT[(i - 1) * m + j];
						y = 0;
						z = 0;

					}
					else {
						x = H_SAT[(i - 1) * m + j];
						y = H_SAT[i * m + (j - 1)];
						z = H_SAT[(i - 1) * m + (j - 1)];

					}

			H_SAT[i * m + j] = image[i * m + j] + y + x - z;

		}

	int indexA = x1 * m + y1;
	int indexD = x2 * m + y2;
	int indexB = x2 * m + y1;
	int indexC = x1 * m + y2;
	float sum = H_SAT[indexA] + H_SAT[indexD] - H_SAT[indexB] - H_SAT[indexC];

	printf("SAT as computed by host\n");  //enable to see HOST SAT
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			printf("%.f ", H_SAT[i * m + j]);
		}
		printf("\n");
	}


	return sum;

}


/*float H_SAT_func(float** image, int img_rows, int img_cols, int x1, int x2, int y1, int y2)
{
	float sum;
	float** SAT;
	SAT = generate_2D_array(img_rows, img_cols);

	int i, j;
	for (j = 0; j < img_cols; j++) 	SAT[0][j] = image[0][j]; //copying first first row


	for (i = 1; i < img_rows; i++)     //column summation
		for (j = 0; j < img_cols; j++)
			SAT[i][j] = image[i][j] + SAT[i - 1][j];

	for (i = 0; i < img_rows; i++)     //row summation
		for (j = 1; j < img_cols; j++)
			SAT[i][j] = SAT[i][j] + SAT[i][j - 1];

	sum = SAT[x2][y2];
	if (x1 > 0) sum = sum - SAT[x1 - 1][y2];
	if (x2 > 0) sum = sum - SAT[x2 - 1][y1];
	if (x1 > 0 && x2 > 0) sum = sum + SAT[x1 - 1][y1 - 1];

	return sum;
}*/

int main()
{
	int x1, y1, x2, y2;
	int N = 0;
	float H_sum;
	float D_sum;
	//int img_rows, img_cols;
	char imagePath[400];

	//printf("please enter the image path\n");
	//scanf("%s", &imagePath);
	printf("how many rectangles?\n");
	scanf("%d", &N);


	//loading image and selecting one channel

	//CImg< unsigned char> image((imagePath));  uncomment to enable arbitrary image
	//int img_cols = image.width();
	//int img_rows = image.height();
	//image.channel(0);

//	float **H_image = generate_2D_array(img_rows, img_cols);
//	float **D_image = generate_2D_array(img_rows, img_cols);

	//copying image pixels to Host image and Device image input matrices
	//int i, j;
	//for (i = 0; i < img_rows; i++)
		//for (j = 0; j < img_cols; j++) {
		//	H_image[i][j] = image(j, i);
		//	D_image[i][j] = image(j, i);
	//	}


	int img_rows = 4; int img_cols = 4;

	float test_image[16] = { 10 ,50, 100 ,0,50 ,255, 35, 0,0, 16, 10, 80,10 ,20, 200, 150 }; // used for testing


	//float **test_image1 = generate_2D_array(4, 4);

	/*for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			test_image1[i][j] = test_image[i * 4 + j];*/


	float* D_SAT = new float[img_rows * img_cols];

	for (int i = 0; i < N; i++)
	{
		printf("please enter the delimiter coordinates(A,D) in order x1, y1, x2, y2 separated by spaces for rectangle no. %d \n", i + 1);

		scanf("%d %d %d %d", &x1, &y1, &x2, &y2);

		clock_t start = clock();
		H_sum = H_SATfunc(test_image, x1, y1, x2, y2, img_rows, img_cols);

		//H_sum = H_SAT_func(test_image1, img_rows, img_cols, x1, x2, y1, y2);

		clock_t stop = clock();
		double H_time = (double)(stop - start) / CLOCKS_PER_SEC;
		printf("\nCPU calculated sum %.4f\n", H_sum);



		start = clock();

		D_sum = D_SATfunc(test_image, D_SAT, x1, y1, x2, y2, img_rows, img_cols);
		stop = clock();
		double D_time = (double)(stop - start) / CLOCKS_PER_SEC;


		printf("SAT as computed by Device \n");  //enable to see SAT


		for (int i = 0; i < img_rows; i++)
		{
			for (int j = 0; j < img_cols; j++)
			{
				printf("%.f ", D_SAT[i * img_cols + j]);
			}
			printf("\n");
		}


		printf("GPU calculated sum %.4f\n", D_sum);

		printf("CPU time %f\n", H_time);

		printf("GPU time %f\n", D_time);

		printf("speedup/slow down achieved with memory overhead = time used by host (%.4f)/ time used by device (%.4f) = %.4f\n", H_time, D_time, H_time / D_time);
		printf("speedup/slow down achieved without memory overhead = time used by host (%.4f)/ time used by kernel (%.4f) = %.4f\n", H_time, kernelTime / 1000.0, H_time / (kernelTime / 1000.0));

	}



	return 0;
}
