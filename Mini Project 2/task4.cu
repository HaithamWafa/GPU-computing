#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include<cuda.h>
#include<iostream>
#include<cmath>
#include<time.h>
#define W1 2000 //width of first matrix
#define C1_W2 1500 //col of first matrix= width of second matrix
#define C2 2300 //col of second matrix
#define TILE_WIDTH 16
double Host_time;
double kernel_time;
double Kmem_time;
using namespace std;

__host__
void matrix_SeqMul(float* h_A, float* h_B, float* h_C, int x1, int x2, int x3)
{

	for (int i = 0; i < x1; i++)
		for (int j = 0; j < x3; j++) {
			float sum = 0;
			for (int k = 0; k < x2; k++) {
				float x = h_A[i *x2 + k];
				float y = h_B[k * x3 + j];
				sum += x * y;
			}
			h_C[i * x3 + j] = sum;
		}
}

__global__
void MatrixMulKernel(float* d_M, float* d_N,float* d_P,int x1, int x2,int x3)
{
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
 
 int bx=blockIdx.x;
 int by=blockIdx.y;
 int tx=threadIdx.x;
 int ty=threadIdx.y;

	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;

for (int i = Row; i < gridDim.y * blockDim.y * 2 + ty; i += gridDim.y * blockDim.y)
 {
	for (int j = Col; j < gridDim.x * blockDim.x * 2 + tx; j += gridDim.x * blockDim.x) 
 {

	float Pvalue = 0;
	for (int ph = 0; ph < (x2+TILE_WIDTH-1) / TILE_WIDTH ; ++ph) {

		if (ph * TILE_WIDTH + tx < x2 && i < x1)
			Mds[ty][tx] = d_M[i * x2 + ph * TILE_WIDTH + tx];
		else
			Mds[ty][tx] = 0.0;

		 
		if (ph * TILE_WIDTH + ty < x2 && j < x3)
			Nds[ty][tx] = d_N[(ph * TILE_WIDTH + ty) * x3 + j];
		else
			Nds[ty][tx] = 0.0;

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH ; k++) {
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}

		__syncthreads();
	}
	
	if (i < x1 && j < x3)
	{
		d_P[i * x3 + j] = Pvalue;
	}

}
 }}


void matrix_ParMul(float* h_A, float* h_B, float* h_C2, int x1, int x2, int x3)
{
	int size_a = x1 * x2 * sizeof(float);
	int size_b = x2 * x3 * sizeof(float);
	int size_c = x1 * x3 * sizeof(float);
	float* d_a, * d_b, * d_c;

	cudaError_t err = cudaMalloc((void**)&d_a, size_a);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_a, h_A, size_a, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void**)&d_b, size_b);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	err = cudaMemcpy(d_b, h_B, size_b, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	err = cudaMalloc((void**)&d_c, size_c);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	
  dim3 dimBlock(TILE_WIDTH , TILE_WIDTH);
	dim3 dimGrid(ceil(x3 / float( dimBlock.x *2)), (ceil(x1 / float(dimBlock.y *2 ))));

	clock_t start = clock();//start
	MatrixMulKernel <<<dimGrid, dimBlock >>> (d_a, d_b, d_c, x1, x2, x3);
	cudaDeviceSynchronize();
 
	clock_t stop = clock();
	kernel_time = (double)(stop - start) / CLOCKS_PER_SEC;
	cout << " Time spent executing kernel code is = " << kernel_time << endl;
 
	err = cudaMemcpy(h_C2, d_c, size_c, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
  {printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);}
 
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

float float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; 
    return min + scale * ( max - min );      
}

void printMatrix2D(float* arr, int rows, int cols) {
for (int i = 0; i < rows; ++i)
		{for (int j = 0; j < cols; ++j)
		{cout<< arr[i * cols + j]<< " ";}
				cout << endl;}
}

int main()
{
printf("For a TILE size of %d x %d\n", TILE_WIDTH, TILE_WIDTH);
time_t t;
srand((unsigned)time(&t));
	
int sizeH_A = W1 * C1_W2;
int sizeH_B = C1_W2 * C2;
int sizeH_C = W1 * C2;

float * h_A = (float *)malloc ( sizeH_A * sizeof(float));
float * h_B = (float *)malloc ( sizeH_B * sizeof(float));
float * h_C = (float *) malloc ( sizeH_C * sizeof(float));
float * h_C2 = (float *) malloc ( sizeH_C * sizeof(float));


for (int i = 0; i < W1; i++)
	for (int j = 0; j < C1_W2; j++)
	{h_A[i * C1_W2 + j] = float_rand(0,100);}
 

for (int i = 0; i < C1_W2; i++)
	for (int j = 0; j < C2; j++)
	{h_B[i * C2 + j] = float_rand(0,100);}
 
clock_t start1 = clock();
matrix_SeqMul(h_A, h_B, h_C, W1, C1_W2, C2);
clock_t stop1 = clock(); //end
Host_time = (double)(stop1 - start1) / CLOCKS_PER_SEC;
cout << "Time spent executing host code is = : " << Host_time << endl;


clock_t start3 = clock();
matrix_ParMul(h_A, h_B, h_C2, W1, C1_W2, C2);
cudaDeviceSynchronize();
clock_t stop3 = clock();
double Kmem_time = (double)(stop3 - start3) / CLOCKS_PER_SEC;
cout << " Time spent executing kernel code using memory is : " << Kmem_time << endl;

/*
cout << "The First Matrix" << endl;
printMatrix2D(h_A, W1, C1_W2) ;
cout << endl;
cout << "The Second Matrix" << endl;
printMatrix2D(h_B, C1_W2, C2) ;
cout << endl;
cout << "Sequential:Multiplication Matrix " << endl;
printMatrix2D(h_C, W1, C2) ;
cout << endl;
cout<< "Parallel:Multiplication Matrix " << endl;
printMatrix2D(h_C2, W1, C2) ;
cout<<endl;*/

 

cout<<"speedup/slow down achieved with memory overhead = time used by host / time used by device  = "<<Host_time/Kmem_time<<endl;
cout<<"speedup/slow down achieved without memory overhead = time used by host / time used by kernel  = "<<Host_time/kernel_time<<endl;
double GFOP = 2*(W1 * C2 / float(pow(10, 9))) * C1_W2;
double H_GFLOPS = GFOP / Host_time;
double D_GFLOPS = GFOP / Kmem_time;
printf("the host was capable of running %.4f GFLOPS, while the device was capable of %.4f GFLOPS\n", H_GFLOPS, D_GFLOPS);
float GPU_util = ((TILE_WIDTH * TILE_WIDTH)/2048.0)*100;
printf("GPU utilization as calculated by number of active threads/SM = %.4f %%\n", GPU_util);

return 0;
}

