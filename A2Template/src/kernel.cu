/*
* Copyright 2024 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms
* is strictly prohibited.
*
*/
#include <helper_cuda.h>
#include "helper.h"



/* Start Header *****************************************************************/

    /*! \file kernal.cu

        \author Ang Jin Yang, jinyang.ang 2000940

        \par email : jinyang.ang.digipen.edu

        \date 2/2/2024

        \brief Copyright (C) 2024 DigiPen Institute of Technology.

    Reproduction or disclosure of this file or its contents without
    the prior written consent of DigiPen Institute of Technology is prohibited. */

/* End Header *******************************************************************/



//P and M column-major, N row-major
__global__ void matrixMultiply(FLOAT_TYPE* P,       //<! [out] and mxn matrix
    const FLOAT_TYPE* M, //<! [in] an mxk matrix k is column
    const FLOAT_TYPE* N, //<! [in] an kxn matrix 
    const int m, const int n, const int k)
{
    // Shared memory for tiling input N array
    __shared__ float N_s[TILE_WIDTH_RATIO_K][TILE_WIDTH_N];

    // Output array variable for P elements
    float P_reg[TILE_WIDTH_N];
    for (int i = 0; i < TILE_WIDTH_N; ++i)
        P_reg[i] = 0.f;


    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int tid = bx * blockDim.x + tx;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int global_index = 0, global_index_M = 0;
    // Loop over the input tiles
    for (int count = 0; count < (k - 1) / TILE_WIDTH_RATIO_K + 1; count++) {

        global_index = (count * TILE_WIDTH_RATIO_K + tx / TILE_WIDTH_N) * n + tidy * TILE_WIDTH_N + tx % TILE_WIDTH_N;
        N_s[tx / TILE_WIDTH_N][tx % TILE_WIDTH_N] = (global_index < k* n) ? N[global_index] : 0.0f;

        // Ensure all threads have finished loading before proceeding
        __syncthreads();

        // Loop over elements inside the tile of N (TILE_WIDTH_RATIO_K iterations)
        for (int elemIdx = 0; elemIdx < TILE_WIDTH_RATIO_K; ++elemIdx) {

            global_index_M = elemIdx * m + (count * TILE_WIDTH_RATIO_K * m) + tid;

            // Load tile of matrix M into register
            float M_reg = (global_index_M < m * k) ? M[global_index_M] : 0.0f;

            // Loop over and update the output elements in P_reg (TILE_WIDTH_N iterations)
            for (int i = 0; i < TILE_WIDTH_N; ++i) {
                P_reg[i] += M_reg * N_s[elemIdx][i];
            }
        }

        // Synchronize threads before proceeding to the next iteration
        __syncthreads();
    }


    for (int i = 0; i < TILE_WIDTH_N; ++i) {

        if (tid < m && (tidy * TILE_WIDTH_N + i) < n)
        {
            P[tid + m * (tidy * TILE_WIDTH_N + i)] = P_reg[i];
        }
    }
    
}

void matrixMultiplyGPU(FLOAT_TYPE* P,
    FLOAT_TYPE* M,
    FLOAT_TYPE* N,
    int numMRows,
    int numNColumns,
    int numMColumns)
{
    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid((numMRows - 1) / TILE_WIDTH_M + 1, (numNColumns - 1) / TILE_WIDTH_N + 1);
    dim3 dimBlock(TILE_WIDTH_M, 1);
    matrixMultiply << <dimGrid, dimBlock >> > (P, M, N, numMRows, numNColumns, numMColumns);

    getLastCudaError("matrixMultiply failed\n");
    cudaDeviceSynchronize();
}
