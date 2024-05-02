/*
* Copyright 2022 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms
* is strictly prohibited.
*
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
/*
* This sample implements Matrix Multiplication
*/

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include "helper.h"

#include <stdint.h>
#include <iostream>
#define epsilon 1.0e-3

void correctness_test(int nRun,
	int numMRows,
	int numMCols,
	int numNCols)
{
	int numNRows = numMCols;
	for (int i = 0; i < nRun; i++) {
		// Call createData() to generate random matrices as inputs
		FLOAT_TYPE* input0 = createDataM(numMRows, numMCols);
		FLOAT_TYPE* input1 = createDataN(numMCols, numNCols);  // Match the dimensions for multiplication
		FLOAT_TYPE* cpuResult = new FLOAT_TYPE[numMRows * numNCols];

		// Matrix multiply using CPU implementation
		matrixMultiplyCPU(cpuResult, input0, input1, numMRows, numMCols, numNCols);

		// Data conversion for GPU
		FLOAT_TYPE* h_M_conv = new FLOAT_TYPE[numMRows * numMCols];
		FLOAT_TYPE* h_P_conv = new FLOAT_TYPE[numMRows * numNCols];
		convertRowColumn(h_M_conv, input0, numMRows, numMCols);

		// Allocate and transfer data to GPU
		FLOAT_TYPE* d_M, * d_N, * d_P;
		checkCudaErrors(cudaMalloc((void**)&d_M, numMRows * numMCols * sizeof(FLOAT_TYPE)));
		checkCudaErrors(cudaMalloc((void**)&d_N, numNRows * numNCols * sizeof(FLOAT_TYPE)));
		checkCudaErrors(cudaMalloc((void**)&d_P, numMRows * numNCols * sizeof(FLOAT_TYPE)));
		checkCudaErrors(cudaMemcpy(d_M, h_M_conv, numMRows * numMCols * sizeof(FLOAT_TYPE), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_N, input1, numNRows * numNCols * sizeof(FLOAT_TYPE), cudaMemcpyHostToDevice));

		// Matrix multiply using GPU implementation
		matrixMultiplyGPU(d_P, d_M, d_N, numMRows, numNCols, numMCols);

		// Transfer the result back to the host
		FLOAT_TYPE* h_P_2 = new FLOAT_TYPE[numMRows * numNCols];
		checkCudaErrors(cudaMemcpy(h_P_2, d_P, numMRows * numNCols * sizeof(FLOAT_TYPE), cudaMemcpyDeviceToHost));
		convertRowColumn(h_P_conv, h_P_2, numNCols, numMRows);

		// Validate GPU result against CPU result
		std::cout << "GPU:" << std::endl;
		for (int i = 0; i < numMRows; ++i) {
			for (int j = 0; j < numNCols; ++j) {
				std::cout << h_P_conv[i * numNCols + j] << " ";
			}
			std::cout << std::endl;
		}

		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "CPU:" << std::endl;

		for (int i = 0; i < numMRows; ++i) {
			for (int j = 0; j < numNCols; ++j) {
				std::cout << cpuResult[i * numNCols + j] << " ";
			}
			std::cout << std::endl;
		}

		bool resultMatch = *cpuResult == *h_P_conv;

		for (int i = 0; i < numMRows; ++i) {
			for (int j = 0; j < numNCols; ++j) {
				if (cpuResult[i * numNCols + j] != h_P_conv[i * numNCols + j])
				{
					resultMatch = false;
					break;
				}
			}
			std::cout << std::endl;
		}

		// Free allocated memory
		delete[] input0;
		delete[] input1;
		delete[] cpuResult;
		delete[] h_M_conv;
		delete[] h_P_conv;
		delete[] h_P_2;
		checkCudaErrors(cudaFree(d_M));
		checkCudaErrors(cudaFree(d_N));
		checkCudaErrors(cudaFree(d_P));
	}
}

void efficiency_test(int nRun,
	int numMRows,
	int numMCols,
	int numNCols)
{
	for (int i = 0; i < nRun; i++) {
		//call createData() to generate random matrix as inputs
		//matrix multiply cpu results
		//measure the time for matrix multiplication cpu version
		//add to total latency for cpu version
		//matrix multiply gpu results
		//measure the time for matrix multiplication gpu version 
		//add to total latency for gpu version
	}
	//average total latency for cpu version over nRun
	//average total latency for gpu version over nRun
}

int main(int argc, char** argv)
{
	int numMRows = 10;
	int numMCols = 10;
	int numNCols = 10;
	int numNRows = numMCols;
	correctness_test(1, numMRows, numMCols, numNCols);

	//correctness_test(1, 101 - rand() % 10, 101 - rand() % 10, 101 - rand() % 10);
	/*correctness_test(1, 200 + rand() % 100, 200 + rand() % 100, 200 + rand() % 100);
	correctness_test(1, 500 + rand() % 500, 500 + rand() % 500, 500 + rand() % 500);
	correctness_test(1, 2000, 2000, 2000);*/

	efficiency_test(10, 100, 100, 100);
	efficiency_test(10, 500, 500, 500);
	efficiency_test(10, 1000, 1000, 1000);

	return 0;
}

