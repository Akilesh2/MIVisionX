#include "rppdefs.h"
#include <omp.h>

RppStatus down_mixing_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  Rpp32s *srcLengthTensor,
                                  Rpp32s *channelsTensor,
                                  bool normalizeWeights)
{
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32s channels = channelsTensor[batchCount];
        Rpp32s samples = srcLengthTensor[batchCount];
        std::vector<float> weights;
        weights.resize(channels, 1.f / channels);
        std::vector<float> normalizedWeights;

        if(normalizeWeights)
        {
            normalizedWeights.resize(channels);

            // Compute sum of the weights
            double sum = 0.0;
            for(int i = 0; i < channels; i++)
                sum += weights[i];

            // Normalize the weights
            float invSum = 1.0 / sum;
            for(int i = 0; i < channels; i++)
                normalizedWeights[i] = weights[i] * invSum;

            weights = normalizedWeights;
        }

        // use weights to downmix to mono
        for(int64_t o = 0, i = 0; o < samples; o++, i += channels)
        {
            dstPtrTemp[o] = 0.0;
            for(int c = 0; c < channels; c++)
                dstPtrTemp[o] += srcPtrTemp[i + c] * weights[c];
        }
    }

    return RPP_SUCCESS;
}
