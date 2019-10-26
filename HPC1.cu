#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cuda_runtime.h>

using namespace std;

__global__ void Min(float* InputArray, int ArraySize){
	int tid = threadIdx.x;
	int ThreadCount = blockDim.x;

	do{
		if(tid<ThreadCount){
			if((tid + ThreadCount)<ArraySize)
				if(InputArray[tid] > InputArray[tid + ThreadCount])
					InputArray[tid] = InputArray[tid + ThreadCount];
		}

		ThreadCount = (ThreadCount+1)>>1;
		ArraySize = (ArraySize + 1)>>1;
	}while(ThreadCount>1);
	
	if(InputArray[0] > InputArray[1])
		InputArray[0] = InputArray[1];
}

__global__ void Max(float* InputArray, int ArraySize){
	int tid = threadIdx.x;
	int ThreadCount = blockDim.x;

	do{
		if(tid<ThreadCount){
			if((tid + ThreadCount)<ArraySize)
				if(InputArray[tid] < InputArray[tid + ThreadCount])
					InputArray[tid] = InputArray[tid + ThreadCount];
		}

		ThreadCount = (ThreadCount+1)>>1;
		ArraySize = (ArraySize + 1)>>1;
	}while(ThreadCount>1);
	
	if(InputArray[0] < InputArray[1])
		InputArray[0] = InputArray[1];
}

__global__ void Sum(float* InputArray, int ArraySize){
	int tid = threadIdx.x;
	int ThreadCount = blockDim.x;

	do{
		if(tid<ThreadCount){
			if((tid + ThreadCount)<ArraySize)
				InputArray[tid] += InputArray[tid + ThreadCount];
		}

		ThreadCount = (ThreadCount+1)>>1;
		ArraySize = (ArraySize + 1)>>1;
	}while(ThreadCount>1);
	
	InputArray[0] += InputArray[1];
}

__global__ void Average(float* InputArray, int ArraySize){
	int tid = threadIdx.x;
	int ThreadCount = blockDim.x;
	int TempArraySize = ArraySize;

	do{
		if(tid<ThreadCount){
			if((tid + ThreadCount)<TempArraySize)
				InputArray[tid] += InputArray[tid + ThreadCount];
		}

		ThreadCount = (ThreadCount+1)>>1;
		TempArraySize = (TempArraySize + 1)>>1;
	}while(ThreadCount>1);
	
	InputArray[0] += InputArray[1];
	
	InputArray[0] /= ArraySize;
}

int main(){
	//Read Array Size From User
	int ArraySize = -1;
	printf("Enter The Number Of Elements: : ");
	scanf("%d", &ArraySize);
	
	if(ArraySize<=0)
		return 0;
	
	//Declare The Float Array
	float *h_Array=new float[ArraySize];

	printf("Enter The Elements In The Array: : ");

	//Read Elements From User
	for(int i=0;i<ArraySize;i++){
		scanf("%f", &h_Array[i]);
	}
		
	int ArrayMemory=ArraySize*sizeof(float);
	int ThreadBlockSize = (ArraySize+1)>>1;
	
	float *d_Array;
	float result;
	
	cudaMalloc(&d_Array,ArrayMemory);
	
	// Copy Array To GPU For Minimum Function
	cudaMemcpy(d_Array, h_Array, ArrayMemory, cudaMemcpyHostToDevice);
  	Min<<<1,ThreadBlockSize>>>(d_Array,ArraySize);
	cudaMemcpy(&result, d_Array, sizeof(float), cudaMemcpyDeviceToHost);
	printf("The Minimum Value In The Array: : %f\n", result);
	
	// Copy Array To GPU For Maximum Function
	cudaMemcpy(d_Array, h_Array, ArrayMemory, cudaMemcpyHostToDevice);
  	Max<<<1,ThreadBlockSize>>>(d_Array,ArraySize);
	cudaMemcpy(&result, d_Array, sizeof(float), cudaMemcpyDeviceToHost);
	printf("The Maximum Value In The Array: : %f\n", result);
	
	// Copy Array To GPU For Sum Function
	cudaMemcpy(d_Array, h_Array, ArrayMemory, cudaMemcpyHostToDevice);
  	Sum<<<1,ThreadBlockSize>>>(d_Array,ArraySize);
	cudaMemcpy(&result, d_Array, sizeof(float), cudaMemcpyDeviceToHost);
	printf("The Sum Of Numbers In The Array: : %f\n", result);
	
	// Copy Array To GPU For Average Function
	cudaMemcpy(d_Array, h_Array, ArrayMemory, cudaMemcpyHostToDevice);
  	Average<<<1,ThreadBlockSize>>>(d_Array,ArraySize);
	cudaMemcpy(&result, d_Array, sizeof(float), cudaMemcpyDeviceToHost);
	printf("The Average Of Numbers In The Array: : %f\n", result);
	
	return 0;
}
