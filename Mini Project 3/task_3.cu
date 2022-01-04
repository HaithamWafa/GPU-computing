#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include<cuda.h>
#include<iostream>
#include<cmath>
#include<time.h>
#define size 1024
#define SM_size 2048
#define N 10
using namespace std;
float product = 0;
float KernelTime,KernelTime1;

__global__ void dotproductKernel(float* d_A, float*d_B,float* d_c, int n){
__shared__ float SM[SM_size];

	int tx = threadIdx.x;
	SM[tx] = 0;
	int bx=blockIdx.x;
	int Dx=blockDim.x;
	int i = bx*Dx + tx;

	if (i < n)
	{SM[tx] = d_A[i] * d_B[i];}
	 
	if (i + 1024 < n)
	{SM[tx + 1024] = d_A[i + 1024] * d_B[i + 1024];}

	__syncthreads();

	for (int j = Dx; j > 0; j >>= 1) 
 {
	if (tx == 0)
  {d_A[bx] = SM[0];}
	if (tx < j) 
 	{SM[tx] += SM[tx + j];}
	__syncthreads();
	}
	d_c[0] = SM[0];
}

double dotProduct(float* h_A,float *h_B, float *h_C,int n)
{

	int size_a = n * sizeof(float);
	int size_b = n * sizeof(float);
	int size_c = ceil(float(n) / size) * sizeof(float);
	float* d_a, * d_b, * d_c;

	cudaError_t err = cudaMalloc(&d_a, size_a);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_a, h_A, size_a, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	 err = cudaMalloc(&d_b, size_b);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_b, h_B, size_b, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);}


	err = cudaMalloc(&d_c, size_c);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaEvent_t start ;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
//clock_t start3 = clock();//start

	dotproductKernel << <1, size >> > (d_a, d_b, d_c, n);

	//clock_t stop3 = clock();
	//KernelTime1 = (double)(stop3 - start3) / CLOCKS_PER_SEC;
	cudaEvent_t stop;
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&KernelTime, start, stop);

	cout<<"Time spent executing kernel code is =" <<KernelTime<<endl;
	
	err = cudaMemcpy(h_C, d_c, size_c, cudaMemcpyDeviceToHost);
	product = h_C[0];   //set by kernel to be SM[0]
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return product;
}

float cpuDotproduct(float *h_a, float* h_b, int n) {
	float total = 0;
	for (int i = 0; i < n; i++)
	{
		total += (h_a[i] * h_b[i]);
	}
	return total;
}

float float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; 
    return min + scale * ( max - min );      
}

void printMatrix2D(float* arr, int rows, int cols) 
{
for (int i = 0; i < rows; ++i)
		{for (int j = 0; j < cols; ++j)
		{cout<< arr[i * cols + j]<< " ";}
		cout << endl;}
}

void printMatrix1D(float* arr, int n)
{
for (int i = 0; i < n; ++i)
{ cout<<arr[i]<<"		";}
cout << endl;
}

int main()
{
	time_t t;
	srand((unsigned)time(&t));

	int sizeH_A = N;
	int sizeH_B = N;
	int sizeH_C = ceil(float (N)/size);

	float * h_A = (float *)malloc ( sizeH_A * sizeof(float));
	float * h_B = (float *)malloc ( sizeH_B * sizeof(float));
	float * h_C = (float *) malloc ( sizeH_C * sizeof(float));
	
	for (int i = 0; i < N; i++)
	{
		h_A[i] = float_rand(0,50);;
		h_B[i] = float_rand(0,50);;
	}

	cout<<"The values of H_a "<<endl;
  printMatrix1D(h_A,N);
	cout<<endl;
	cout<<"The values of H_b "<<endl;
	printMatrix1D(h_B,N);
	cout<<endl;

	clock_t start0 = clock();
 	float dot_product=cpuDotproduct(h_A,h_B,N);
	clock_t stop0 = clock();
	double Host_time = (double)(stop0 - start0) / CLOCKS_PER_SEC;

	cout << "Sequential dot product =" << dot_product << endl;
	cout << "Time spent executing host code is = : " << Host_time << endl;
	

	clock_t start1 = clock();
	float product =dotProduct(h_A,h_B, h_C,N);
	clock_t	stop1 = clock();

	double Kmem_time = (double)(stop1 - start1) / CLOCKS_PER_SEC;
	cout << "Parallel dot product =" << product << endl;
	cout << "Time spent executing kernel code using memory is : " << Kmem_time << endl;

	cout<<"speedup/slow down achieved with memory overhead = time used by host / time used by device  = "<<Host_time/Kmem_time<<endl;
	cout<<"speedup/slow down achieved without memory overhead = time used by host / time used by kernel  = "<<Host_time/KernelTime<<endl;
	return 0;
}
