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
#include <stdlib.h>
#include "helper.h"
FLOAT_TYPE* createDataM(int nRows, int nCols)
{
	FLOAT_TYPE* data = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * nCols * nRows);
	int i;
	for (i = 0; i < nCols * nRows; i++) {
		data[i] = ((FLOAT_TYPE)(rand() % 10) - 5) / 5.0f;
		//data[i] = 5.0f;
	}
	return data;
}

FLOAT_TYPE* createDataN(int nRows, int nCols)
{
	FLOAT_TYPE* data = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * nCols * nRows);
	int i;
	for (i = 0; i < nCols * nRows; i++) {
		data[i] = ((FLOAT_TYPE)(rand() % 10) - 5) / 5.0f;
		//data[i] = 2.0f;
	}
	return data;
}

void matrixMultiplyCPU(FLOAT_TYPE* output, FLOAT_TYPE* input0, FLOAT_TYPE* input1,
	int numMRows, int numMColumns, int numNColumns)
{
	for (int i = 0; i < numMRows; ++i) {
		for (int j = 0; j < numNColumns; ++j) {
			FLOAT_TYPE sum = 0.0;
			for (int k = 0; k < numMColumns; ++k) {
				sum += input0[i * numMColumns + k] * input1[k * numNColumns + j];
			}
			output[i * numNColumns + j] = sum;
		}
	}
}

void convertRowColumn(FLOAT_TYPE* dst, FLOAT_TYPE* src, int numRows, int numCols)
{
	for (int i = 0; i < numRows; ++i) {
		for (int j = 0; j < numCols; ++j) {
			dst[j * numRows + i] = src[i * numCols + j];
		}
	}
}