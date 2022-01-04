#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdlib.h>
#include <stdio.h>
#include <iostream>
# define N 5
using namespace std;

__global__ void Gauss_kernel(float *d_a , float *d_b ,int n) 
{
 
    int idx = threadIdx.x ; 
    int idy = threadIdx.y ; 
 
    for(int i =1 ; i<n ;i++) 
    { 
        if((idy + i) < n) 
        { 
          float part1=d_a[((i-1) * (n+1)) + (i-1)] ;
          float part2=d_a[((i+idy) * (n+1)) + (i-1)] ;
          float var1 =(-1)*( part1/part2); 
          float p1=d_a[((i-1) * (n+1)) + (idx)] ;
          float p2=(var1) * d_a[((i+idy) * (n+1)) + (idx)] ;
          d_a[((i+idy) * (n+1)) + (idx)]  = p1 +p2;
        } 
        __syncthreads(); 
    } 
    
    d_b[idy*(n+1) + idx] = d_a[((idy) * (n+1)) + (idx)] ;
}

void GaussElimination(float *H_A , int n , float *H_B) 
{ 
      int size_a = N * (N+1) * sizeof(float);
      int size_b = N * (N+1) * sizeof(float);
      float* d_a, * d_b ;
     
      cudaError_t err = cudaMalloc((void**)&d_a, size_a);
      if (err != cudaSuccess)
      {printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);}

      err = cudaMemcpy(d_a, H_A, size_a, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) 
      {printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);}

      err = cudaMalloc((void**)&d_b, size_b);
      if (err != cudaSuccess) 
      {printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);}
 

    dim3 dimBlock(n+1,n,1); 
    dim3 dimGrid(1,1,1); 
    Gauss_kernel<<<dimGrid , dimBlock>>>(d_a , d_b , n);
    
    err = cudaMemcpy(H_B, d_b, size_b, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {printf("%s in %s at line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
      exit(EXIT_FAILURE);}

    cudaFree(d_a); 
    cudaFree(d_b); 
}

void printMatrix2D(float* arr, int rows, int cols) 
{
for (int i = 0; i < rows; ++i)
    {for (int j = 0; j < cols; ++j)
    {cout<< arr[i * cols + j]<< "\t";}
        cout << endl;}
}

float float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; 
    return min + scale * ( max - min );      
}

int main() 
{   time_t t;
    srand((unsigned)time(&t));
    int sizeH_A =N * (N+1);
    int sizeH_B = N * (N+1);
    int size_result =N;
    float *H_A = (float*)malloc(sizeH_A * sizeof(float));
    float *H_B = (float*)malloc(sizeH_B* sizeof(float));
    float *final = (float*)malloc(size_result*sizeof(float));
    
    

for (int i = 0; i < N; i++)
	for (int j = 0; j < N+1; j++)
	{H_A[i * N+1 + j] = float_rand(0,5);}

 
    GaussElimination(H_A , N , H_B); 
 
    cout << "The Matrix initially is" << endl;
    printMatrix2D(H_A, N, N+1) ;
    cout<<endl;
    cout << "The Matrix H_B is" << endl;
    printMatrix2D(H_B, N, N+1) ;


    for(int i=N-1 ; i>=0 ; i--) 
    {  
      float Total = 0.0 ;
      float temp=0;
      int j;
      for(j=N-1 ; j>i ;j--) 
      { Total = Total + final[j]*H_B[i*(N+1) + j]; 
       //printf("sum at i= %d, j=%d is  =%f\n",i,j,Total);
      }
      temp = H_B[i*(N+1) + N] - Total ; 
      final[i] = temp / H_B[i *(N+1) + j];
     } 
    
    cout<<endl;
    printf("Final value of each variable\n"); 
    for(int i =0;i<N;i++) 
    { printf(" variable_%d = %f\n", i ,final[i]); } 
    return 0; 
}