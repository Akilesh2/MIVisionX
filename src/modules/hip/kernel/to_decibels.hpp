#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void to_decibels_hip_compute(d_float8 *src_f8, d_float8 *dst_f8, float minRatio, float multiplier, float invReferenceMagnitude)
{
    dst_f8->f1[0] = multiplier * log10f(fmaxf(minRatio, src_f8->f1[0] * invReferenceMagnitude));
    dst_f8->f1[1] = multiplier * log10f(fmaxf(minRatio, src_f8->f1[1] * invReferenceMagnitude));
    dst_f8->f1[2] = multiplier * log10f(fmaxf(minRatio, src_f8->f1[2] * invReferenceMagnitude));
    dst_f8->f1[3] = multiplier * log10f(fmaxf(minRatio, src_f8->f1[3] * invReferenceMagnitude));
    dst_f8->f1[4] = multiplier * log10f(fmaxf(minRatio, src_f8->f1[4] * invReferenceMagnitude));
    dst_f8->f1[5] = multiplier * log10f(fmaxf(minRatio, src_f8->f1[5] * invReferenceMagnitude));
    dst_f8->f1[6] = multiplier * log10f(fmaxf(minRatio, src_f8->f1[6] * invReferenceMagnitude));
    dst_f8->f1[7] = multiplier * log10f(fmaxf(minRatio, src_f8->f1[7] * invReferenceMagnitude));
}

__global__ void to_decibels_tensor(float *srcPtr,
                                   uint2 srcStridesNH,
                                   float *dstPtr,
                                   uint2 dstStridesNH,
                                   int *srcLengthTensor,
                                   float minRatio,
                                   float multiplier,
                                   float referenceMagnitude,
                                   float *maxValues)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= srcLengthTensor[id_z])
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + id_x;
    referenceMagnitude = (referenceMagnitude == 0.0) ? maxValues[id_z] : referenceMagnitude;
    float invreferenceMagnitude = (1.0f / referenceMagnitude);

    d_float8 src_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    to_decibels_hip_compute(&src_f8, &dst_f8, minRatio, multiplier, invreferenceMagnitude);

    uint dstIdx = (id_z * dstStridesNH.x) + id_x;
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}


__global__ void get_max(float *srcPtr,
                        int *srcLength,
                        uint1 srcStride,
                        float *max,
                        int *mutex)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int N = srcLength[id_z];

    if(id_x >= N)
        return;

    uint srcIdx = (id_z * srcStride.x) + id_x;
    int threadIdx = hipThreadIdx_x;
    int blockDimx = hipBlockDim_x;

    // Store block data in shared memory
    extern __shared__ float shm[];
    shm[threadIdx] = srcPtr[srcIdx];
    __syncthreads();

    // Do reduction
    for(int s = 1; s < hipBlockDim_x; s = s * 2)
    {
        int loc = 2 * s * threadIdx;
        if((loc + s) < N)
            shm[loc] = fmaxf(shm[loc], shm[loc + s]);

        __syncthreads();
    }

    // Get global maximum across blocks
    if(threadIdx == 0)
    {
		while(atomicCAS(mutex, 0, 1) != 0);  // lock
		max[id_z] = fmaxf(max[id_z], shm[0]);
		atomicExch(mutex, 0);  // unlock
	}
}

RppStatus hip_exec_to_decibels_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcLengthTensor,
                                      Rpp32f cutOffDB,
                                      Rpp32f multiplier,
                                      Rpp32f referenceMagnitude,
                                      rpp::Handle& handle)
{
    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Z;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = 1;
    int globalThreads_z = handle.GetBatchSize();

    if(referenceMagnitude == 0.0)
    {
        int *mutex = nullptr;
        hipMalloc((void **)&mutex, sizeof(int));
        hipMemset(handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem, -std::numeric_limits<float>::max(), handle.GetBatchSize() * sizeof(float));
        hipMemset(mutex, 0, sizeof(int));

        globalThreads_x = srcDescPtr->strides.hStride;
        hipLaunchKernelGGL(get_max,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           LOCAL_THREADS_X,
                           handle.GetStream(),
                           srcPtr,
                           srcLengthTensor,
                           make_uint1(srcDescPtr->strides.nStride),
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           mutex);
        hipDeviceSynchronize();
        hipFree(mutex);
    }

    float minRatio = powf(10, cutOffDB / multiplier);
    globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
    hipLaunchKernelGGL(to_decibels_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                       dstPtr,
                       make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                       srcLengthTensor,
                       minRatio,
                       multiplier,
                       referenceMagnitude,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem);

    return RPP_SUCCESS;
}