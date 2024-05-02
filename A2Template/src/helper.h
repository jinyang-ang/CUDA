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

#ifndef HELPER_H
#define HELPER_H

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define UINT_BITS 32

typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned char uchar;
typedef float FLOAT_TYPE;
//typedef double FLOAT_TYPE;

#define TILE_WIDTH_M 8//64
#define TILE_WIDTH_N 2//16
#define TILE_WIDTH_RATIO_K (TILE_WIDTH_M / TILE_WIDTH_N)

//create random matrix
extern "C" FLOAT_TYPE* createDataM(int nRow, int nCols);
extern "C" FLOAT_TYPE * createDataN(int nRow, int nCols);

//convert rows into columns or convert columns into rows for matrix
extern "C" void convertRowColumn(FLOAT_TYPE* dst, FLOAT_TYPE* src, int numRows, int numCols);

////////////////////////////////////////////////////////////////////////////////
// CPU version for Matrix Multiplication
////////////////////////////////////////////////////////////////////////////////
extern "C" void matrixMultiplyCPU(FLOAT_TYPE* output, FLOAT_TYPE* input0, FLOAT_TYPE* input1,
	int numMRows, int numMColumns, int numNColumns);

////////////////////////////////////////////////////////////////////////////////
// GPU version for Matrix Multiplication
////////////////////////////////////////////////////////////////////////////////
extern "C" void matrixMultiplyGPU(FLOAT_TYPE* P,
	FLOAT_TYPE* M,
	FLOAT_TYPE* N,
	int numMRows,
	int numNColumns,
	int numMColumns);

#endif
