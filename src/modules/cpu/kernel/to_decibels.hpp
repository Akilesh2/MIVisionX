#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus to_decibels_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptImagePatchPtr srcDims,
                                  Rpp32f cutOffDB,
                                  Rpp32f multiplier,
                                  Rpp32f referenceMagnitude)
{
    bool referenceMax = (referenceMagnitude == 0.0) ? false : true;

    // Calculate the intermediate values needed for DB conversion
    Rpp32f minRatio = std::pow(10, cutOffDB / multiplier);
    if(minRatio == 0.0f)
        minRatio = std::nextafter(0.0f, 1.0f);

    Rpp32f log10Factor = 1 / std::log(10);
    multiplier *= log10Factor;

    __m256 pMultiplier = _mm256_set1_ps(multiplier);
    __m256 pMinRatio = _mm256_set1_ps(minRatio);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrCurrent = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrCurrent = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32u height = srcDims[batchCount].height;
        Rpp32u width = srcDims[batchCount].width;

        // Compute maximum value in the input buffer
        if(!referenceMax)
            referenceMagnitude = *(std::max_element(srcPtrCurrent, srcPtrCurrent + (height * width)));

        // Avoid division by zero
        if(referenceMagnitude == 0.0)
            referenceMagnitude = 1.0;

        Rpp32f invReferenceMagnitude = 1.f / referenceMagnitude;
        __m256 pinvMag = _mm256_set1_ps(invReferenceMagnitude);

        int vectorIncrement = 8;
		int alignedLength = (width / 8) * 8;
		int vectorLoopCount = 0;

        Rpp32f *srcPtrRow, *dstPtrRow;
        srcPtrRow = srcPtrCurrent;
        dstPtrRow = dstPtrCurrent;
        for(int i = 0; i < height; i++)
        {
            Rpp32f *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrRow;
            dstPtrTemp = dstPtrRow;
            for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
            {
                __m256 pSrc;
                pSrc = _mm256_loadu_ps(srcPtrTemp);
                pSrc = _mm256_mul_ps(pSrc, pinvMag);
                pSrc = _mm256_max_ps(pMinRatio, pSrc);
                pSrc = log_ps(pSrc);
                pSrc = _mm256_mul_ps(pSrc, pMultiplier);
                _mm256_storeu_ps(dstPtrTemp, pSrc);
                srcPtrTemp += vectorIncrement;
                dstPtrTemp += vectorIncrement;
            }
            for(; vectorLoopCount < width; vectorLoopCount++)
            {
                *dstPtrTemp = multiplier * std::log(std::max(minRatio, (*srcPtrTemp) * invReferenceMagnitude));
                srcPtrTemp++;
                dstPtrTemp++;
            }

            srcPtrRow += srcDescPtr->strides.hStride;
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    }

    return RPP_SUCCESS;
}