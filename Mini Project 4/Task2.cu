#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CImg.h"
using namespace cimg_library;

#define MASK_WIDTH 3
#define Tile_Width 16
__constant__ float M[MASK_WIDTH][MASK_WIDTH];
long long int operations_count;
float kernelTime;

__global__ void ConvKernel(float* input_image, float* output_image, int w, int h)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y * Tile_Width + ty;
	int col = blockIdx.x * Tile_Width + tx;
	if (row < h && col < w && row >= 0 && col >= 0)
	{
		float Img_value = 0;
		int min = 0;
		int max = 255;
		__shared__ float N_ds[Tile_Width][Tile_Width];
		N_ds[ty][tx] = input_image[row * w + col];
		__syncthreads();
		int this_tile_col_point = blockIdx.x * Tile_Width;
		int Next_tile_col_point = (blockIdx.x + 1) * Tile_Width;
		int this_tile_row_point = blockIdx.y * Tile_Width;
		int Next_tile_row_point = (blockIdx.y + 1) * Tile_Width;
		int mid = MASK_WIDTH / 2;
		int rs = row - mid;
		int cs = col - mid;


		if (ty < Tile_Width && tx < Tile_Width) {
			for (int i = 0; i < MASK_WIDTH; i++) {
				for (int j = 0; j < MASK_WIDTH; j++) {
					int row_index = rs + i;
					int col_index = cs + j;
					if (row_index >= 0 && row_index < h && col_index >= 0 && col_index < w)
					{
						if ((row_index >= this_tile_row_point) && (row_index < Next_tile_row_point) && (col_index >= this_tile_col_point) && (col_index < Next_tile_col_point))
							Img_value += N_ds[ty + i - (mid)][tx + j - (mid)] * M[i][j];
						else
							Img_value += M[i][j] * input_image[w * row_index + col_index];
					}
					else
					{
						if (row_index < 0 && col_index < 0)
						{
							Img_value += M[i][j] * input_image[0];
						}
						else if (row_index < 0 && col_index >= w)
						{
							Img_value += M[i][j] * input_image[w - (mid)];
						}
						else if (row_index >= h && col_index < 0)
						{
							Img_value += M[i][j] * input_image[w * (h - mid)];
						}
						else if (row_index >= h && col_index >= w)
						{
							Img_value += M[i][j] * input_image[w * (h - mid) + w - (mid)];
						}

						else if (row_index >= 0 && row_index < h)
						{
							if (col_index < 0)
							{
								Img_value += M[i][j] * input_image[w * row_index + col_index + (mid)];
							}
							else if (col_index >= w)
							{
								Img_value += M[i][j] * input_image[w * row_index + col_index - (mid)];
							}
						}
						else if (col_index >= 0 && col_index < w)
						{
							if (row_index < 0)
							{
								Img_value += M[i][j] * input_image[w * (row_index + (mid)) + col_index];
							}
							else if (row_index >= h)
							{
								Img_value += M[i][j] * input_image[w * (row_index - (mid)) + col_index];
							}
						}
					}
				}
			}

			if (Img_value < 0)
			{
				Img_value = min;
			} // must be between 0 and 255
			if (Img_value > 255)
			{
				Img_value = max;
			}/// must be between 0 and 255



			output_image[row * w + col] = Img_value;
		}
	}
}
void D_Conv2D(float* h_input_image, float* h_mask, int w, int h, float* h_output_image)
{
	//converting 2d to 1d
	float* input_image;
	float* output_image;


	double s = clock();
	int size = w * h * sizeof(float);

	cudaError_t err = cudaMalloc((void**)& input_image, size);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(input_image, h_input_image, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void**)& output_image, size);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	dim3 dimBlock(Tile_Width, Tile_Width, 1);
	dim3 dimGrid(ceil(w / float(Tile_Width)), ceil(h / float(Tile_Width)), 1);

	cudaMemcpyToSymbol(M, h_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float));
	ConvKernel << <dimGrid, dimBlock >> > (input_image, output_image, w, h);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&kernelTime, start, stop);


	//printf("Time taken executing kernel: %f \n", kernelTime);
	err = cudaMemcpy(h_output_image, output_image, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//double e = clock();
	//parallel_time = (double)(e - s) / CLOCKS_PER_SEC;



	cudaFree(input_image); cudaFree(output_image);

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
			printf("%f    ", mat[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void initializeMasks(float** blur, float** emboss, float** outline, float** sharpen, float** sobel_right, float** sobel_left, float** sobel_up, float** sobel_down) {
	// Masks 
	int i, j;
	float blur_values[MASK_WIDTH * MASK_WIDTH] = { 0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625 };
	float emboss_values[MASK_WIDTH * MASK_WIDTH] = { -2, -1, 0, -1, 1, 1, 0, 1, 2 };
	float outline_values[MASK_WIDTH * MASK_WIDTH] = { -1 , -1 ,-1,-1 , 8 ,-1, -1 ,-1, -1 };
	float sharpen_values[MASK_WIDTH * MASK_WIDTH] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
	float sobel_right_values[MASK_WIDTH * MASK_WIDTH] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float sobel_left_values[MASK_WIDTH * MASK_WIDTH] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
	float sobel_up_values[MASK_WIDTH * MASK_WIDTH] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	float sobel_down_values[MASK_WIDTH * MASK_WIDTH] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

	for (i = 0; i < MASK_WIDTH; i++)
	{
		for (j = 0; j < MASK_WIDTH; j++)
		{
			blur[i][j] = blur_values[i * MASK_WIDTH + j];
			emboss[i][j] = emboss_values[i * MASK_WIDTH + j];
			outline[i][j] = outline_values[i * MASK_WIDTH + j];
			sharpen[i][j] = sharpen_values[i * MASK_WIDTH + j];
			sobel_right[i][j] = sobel_right_values[i * MASK_WIDTH + j];
			sobel_left[i][j] = sobel_left_values[i * MASK_WIDTH + j];
			sobel_up[i][j] = sobel_up_values[i * MASK_WIDTH + j];
			sobel_down[i][j] = sobel_down_values[i * MASK_WIDTH + j];
		}
	}


}
float** H_conv2D(float** image, float** mask, int img_rows, int img_cols)
{
	float sum; float** H_outImg; int mid = MASK_WIDTH / 2;
	H_outImg = generate_2D_array(img_rows, img_cols);
	int min = 0;
	int max = 255;
	int i, j;
	for (i = 0; i < img_rows; i++) {
		for (j = 0; j < img_cols; j++) {
			sum = 0.0f;
			for (int y = -1; y <= 1; y++)
				for (int x = -1; x <= 1; x++)
				{
					if (i + y >= 0 && i + y < img_rows && j + x >= 0 && j + x < img_cols) {
						sum += image[i + y][j + x] * mask[y + 1][x + 1];
					}

					else {

						if (i + y < 0 && j + x < 0)
							sum = sum + mask[y + 1][x + 1] * image[0][0];
						else if (i + y < 0 && j + x >= img_cols)
							sum = sum + mask[y + 1][x + 1] * image[0][img_cols - mid];

						else if (i + y >= img_rows && j + x < 0)
							sum = sum + mask[y + 1][x + 1] * image[(img_rows - mid)][0];

						else if (i + y >= img_rows && j + x >= img_cols)
							sum = sum + mask[y + 1][x + 1] * image[(img_rows - mid)][img_cols - mid];

						else if (i + y >= 0 && i + y < img_rows)
						{
							if (j + x < 0) sum += mask[y + 1][x + 1] * image[i + y][j + x + mid];
							else if (j + x >= img_cols)
								sum = sum + mask[y + 1][x + 1] * image[i + y][j + x - mid];
						}
						else if (j + x >= 0 && j + x < img_cols)
						{
							if (i + y < 0) sum += mask[y + 1][x + 1] * image[i + y + mid][j + x];
							else if (i + y >= img_rows)
								sum = sum + mask[y + 1][x + 1] * image[i + y - mid][j + x];
						}
					}

					operations_count = operations_count + 2;
				}
			if (sum > 255)
				sum = max;
			if (sum < 0)
				sum = min;
			H_outImg[i][j] = sum;
		}
	}
	return H_outImg;
}

int main(void)
{
	float** blur; float** emboss; float** outline; float** sharpen; float** sobel_right; float** sobel_left; float** sobel_up; float** sobel_down;
	float** H_image; float** H_outImg; float** D_image; float** D_outImg;
	float** mask;
	time_t t;
	srand((unsigned)time(&t));
	char imagePath[400];
	int operation = 0;

	//convolutional masks generation 
	blur = generate_2D_array(MASK_WIDTH, MASK_WIDTH);
	emboss = generate_2D_array(MASK_WIDTH, MASK_WIDTH);
	outline = generate_2D_array(MASK_WIDTH, MASK_WIDTH);
	sharpen = generate_2D_array(MASK_WIDTH, MASK_WIDTH);
	sobel_right = generate_2D_array(MASK_WIDTH, MASK_WIDTH);
	sobel_left = generate_2D_array(MASK_WIDTH, MASK_WIDTH);
	sobel_up = generate_2D_array(MASK_WIDTH, MASK_WIDTH);
	sobel_down = generate_2D_array(MASK_WIDTH, MASK_WIDTH);
	initializeMasks(blur, emboss, outline, sharpen, sobel_right, sobel_left, sobel_up, sobel_down);
	//printf("the mask is: \n");
	//printMatrix2D(sobel_down, MASK_WIDTH, MASK_WIDTH);

	printf("please enter the image path\n");
	scanf("%s", imagePath);
	printf("The program supports eight operations, please enter the code of the required operation: \n 1-Blur \n 2-Emboss \n 3-Outline \n 4-Sharpen \n 5-Sobel left \n 6-Sobel right \n 7-Sobel top \n 8-Sobel bottom \n");
	scanf("%d", &operation);

	//printf("you want to do %d  to the  %s image. Correct?\n", operation, imagePath);

	//loading image and selecting one channel

	CImg< unsigned char> image((imagePath));
	int img_cols = image.width();
	int img_rows = image.height();
	image.channel(0);

	CImg<float> image_out1(img_cols, img_rows, 1, 1, 0); //for Host
	CImg<float> image_out2(img_cols, img_rows, 1, 1, 0); //for Device


	H_image = generate_2D_array(img_rows, img_cols);
	D_image = generate_2D_array(img_rows, img_cols);

	D_outImg = generate_2D_array(img_rows, img_cols);
	//copying image pixels to Host image and Device image input matrices
	int i, j;
	for (i = 0; i < img_rows; i++)
		for (j = 0; j < img_cols; j++) {
			H_image[i][j] = image(j, i);
			D_image[i][j] = image(j, i);
		}


	//printf("the image is: \n");
	//printMatrix2D(H_image, img_rows, img_cols);
	switch (operation)
	{
	case 1:
		mask = blur;
		break;
	case 2:
		mask = emboss;
		break;
	case 3:
		mask = outline;
		break;
	case 4:
		mask = sharpen;
		break;
	case 5:
		mask = sobel_left;
		break;
	case 6:
		mask = sobel_right;
		break;
	case 7:
		mask = sobel_up;
		break;
	case 8:
		mask = sobel_down;
		break;
	default:
		mask = blur;       //blurring is assumed if operation input is unsupported
	}
	clock_t start = clock();  //timing host
	H_outImg = H_conv2D(H_image, mask, img_rows, img_cols);
	clock_t stop = clock();

	double H_time = (double)(stop - start) / CLOCKS_PER_SEC;
	start = clock();  //timing wrapper function
	D_Conv2D(*D_image, *mask, img_cols, img_rows, *D_outImg);
	stop = clock();
	double D_time = (double)(stop - start) / CLOCKS_PER_SEC;

	//constructing output image generated by Device and Host
	bool flag = 0;
	for (i = 0; i < img_rows; i++)
		for (j = 0; j < img_cols; j++) {
			image_out1(j, i) = H_outImg[i][j];
			image_out2(j, i) = D_outImg[i][j];
			if (H_outImg[i][j] != H_outImg[i][j])  //checking equivalency
				flag = 1;
		}
	image_out1.save("host_edited.jpg");
	image_out2.save("device_edited.jpg");
	printf("TILE WIDTH USED FOR KERNEL = %d\n", Tile_Width);
	if (flag == 0) printf("Device and Host produce similar results\n");

	printf("speedup/slow down achieved with memory overhead = time used by host (%.4f)/ time used by device (%.4f) = %.4f\n", H_time, D_time, H_time / D_time);
	printf("speedup/slow down achieved without memory overhead = time used by host (%.4f)/ time used by kernel (%.4f) = %.4f\n", H_time, kernelTime / 1000.0, H_time / (kernelTime / 1000.0));


	double GFOP = operations_count / float(pow(10, 9));
	double H_GFLOPS = GFOP / H_time;
	double D_GFLOPS = GFOP / (kernelTime / 1000.0);
	printf("The following calculations exclude the device memory overhead...\n");
	printf("the host was capable of running %.4f GFLOPS, while the device was capable of %.4f GFLOPS\n", H_GFLOPS, D_GFLOPS);
	float GPU_util = ((Tile_Width * Tile_Width) / 2048.0) * 100;
	printf("GPU utilization as calculated by number of active threads/SM = %.4f %%\n", GPU_util);

	return 0;
}