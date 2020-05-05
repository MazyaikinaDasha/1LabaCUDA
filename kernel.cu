#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include "device_launch_parameters.h"

#define BLOCK_SIZE 32

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);


__global__ void Mult(float* a, float* b, int n, float* c)
{
	// ����� �����
	int bx = blockIdx.x; 
	int by = blockIdx.y; 
	// ����� ���� 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 

	float sum = 0.0f;
	// ����� ������ �� A
	int ia = n * (BLOCK_SIZE * by + ty); 
	// ����� ������� �� B
	int ib = BLOCK_SIZE * bx + tx; 
	// ����� �������� �� �
	int ic = ia + ib; 
	for (int k = 0; k < n; k++) {
		sum += a[ia + k] * b[ib + k * n];
	}
	c[ic] = sum;
}

int main()
{
	int N = 2048;
	int m, n, k;
	// �������� ����������-�������
	float timerValueGPU, timerValueCPU = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	int numBytes = N * N * sizeof(float);
	float* adev, * bdev, * cdev, * a, * b, * c, * cc;
	//������� A
	a = (float*)malloc(numBytes); 
	//������� B
	b = (float*)malloc(numBytes); 
	//������� � ��� GPU-��������
	c = (float*)malloc(numBytes); 
	//������� � ��� CPU-��������
	cc = (float*)malloc(numBytes); 

	// ������� ������� A, B � ����������������� ������� B
	for (n = 0; n < N; n++)
	{
		for (m = 0; m < N; m++)
		{
			a[m + n * N] = 2.0f * m + n; b[m + n * N] = m - n;
		}
	}
	// ������� ����� ����� � ������
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);
	// ��������� ������ �� GPU
	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);


	// GPU-������� 
	// ����������� ������ A � B � host �� device
	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	// ������ �������-����
	Mult << < blocks, threads >> > (adev, bdev, N, cdev);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time %f msec\n", timerValueGPU);
	// �����������, ����������� ������� C � device �� host
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

	//  CPU-������� 
	double start_time = clock();
	// ���������� ������� C
	for (int i = 0; i < N; i++) {
		for (int k = 0; k < N; k++) {
			for (int j = 0; j < N; j++) {
				cc[i * N + j] += a[i * N + k] * b[k * N + j];
			}
		}
	}
	double end_time = clock();
	timerValueCPU = ((end_time - start_time)) * 1000 / CLOCKS_PER_SEC;

	printf("\n CPU calculation time %f msec\n", timerValueCPU);
	printf("\n Rate %f x\n", timerValueCPU / timerValueGPU);

	// ������������ ������ �� GPU � CPU
	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);
	free(a);
	free(b);
	free(c);
	free(cc);
	// ����������� ����������-�������
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	system("pause");
	return 0;

}