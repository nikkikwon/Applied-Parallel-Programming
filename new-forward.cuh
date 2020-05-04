
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 16
namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

    int b = blockIdx.z;
    int h_t, w_t, c_t, p_t, q_t;

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    int numAColumns = C * K * K;
    int numARows = M;
    int numBColumns = H_out * W_out;
    int numBRows = C * K * K;
    int numCColumns = numBColumns;
    int numCRows = numARows;
    
    float Pvalue = 0;


// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]


    for (int i = 0; i < (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
        if (i * TILE_WIDTH + tx < numAColumns && Row < numARows) {
            subTileM[ty][tx] = k[Row * numAColumns + i * TILE_WIDTH + tx];
        } else {
            subTileM[ty][tx] = 0.0;
        }
        if (Col < numBColumns && i * TILE_WIDTH + ty < numBRows) {
            h_t = Col / W_out;
            w_t = Col % W_out;
            c_t = (i * TILE_WIDTH + ty) / (K * K);
            p_t = (i * TILE_WIDTH + ty) % (K * K) / K;
            q_t = (i * TILE_WIDTH + ty) % (K * K) % K;
            subTileN[ty][tx] = x4d(b, c_t, h_t + p_t, w_t + q_t);
        } else {
            subTileN[ty][tx] = 0.0;
        }
        __syncthreads();
        if (Row < numCRows && Col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) 
                Pvalue += subTileM[ty][i] * subTileN[i][tx];
        }
        __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns) {
        y[(b * M * H_out * W_out) + Row * numCColumns + Col] = Pvalue;
    }

#undef x4d

}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
        
    // Set the kernel dimensions
    dim3 gridDim(ceil((1.0 * H_out * W_out)/TILE_WIDTH), ceil((1.0 * M)/TILE_WIDTH), B);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH, 1);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif