/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "internal_publishKernels.h"
#include <chrono>
#include <utility>
long long unsigned rpp_brightness_time = 0;
long long unsigned brightness_refresh_time = 0;
long long unsigned brightness_init_time = 0;
long long unsigned brightness_uninit_time = 0;
long long unsigned brightness_complete_process_time = 0;
#define TENSOR_RPP 1
#define PROCESS_TIME_LOG 0
#define LOG_LEVEL_2 0

struct BrightnessbatchPDLocalData
{
    RPPCommonHandle handle;
    rppHandle_t rppHandle;
    Rpp32u device_type;
    Rpp32u nbatchSize;
    RppiSize *srcDimensions;
    RppiSize maxSrcDimensions;
    Rpp32u *srcBatch_width;
    Rpp32u *srcBatch_height;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    vx_float32 *alpha;
    vx_float32 *beta;
    #if TENSOR_RPP
        RpptDescPtr srcDescPtr, dstDescPtr;
        RpptROIPtr roiTensorPtrSrc;
        RpptRoiType roiType;
        RpptDesc srcDesc, dstDesc;
    #endif
#if ENABLE_OPENCL
    cl_mem cl_pSrc;
    cl_mem cl_pDst;
#elif ENABLE_HIP
    void *hip_pSrc;
    void *hip_pDst;
    #if TENSOR_RPP
        RpptROI *d_roiTensorPtrSrc;
    #endif
#endif
};

static vx_status VX_CALLBACK refreshBrightnessbatchPD(vx_node node, const vx_reference *parameters, vx_uint32 num, BrightnessbatchPDLocalData *data)
{
    vx_status status = VX_SUCCESS;
    vx_status copy_status;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, data->nbatchSize, sizeof(vx_float32), data->alpha, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, data->nbatchSize, sizeof(vx_float32), data->beta, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &data->maxSrcDimensions.height, sizeof(data->maxSrcDimensions.height)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &data->maxSrcDimensions.width, sizeof(data->maxSrcDimensions.width)));
    data->maxSrcDimensions.height = data->maxSrcDimensions.height / data->nbatchSize;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[1], 0, data->nbatchSize, sizeof(Rpp32u), data->srcBatch_width, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[2], 0, data->nbatchSize, sizeof(Rpp32u), data->srcBatch_height, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    for (int i = 0; i < data->nbatchSize; i++)
    {
        #if TENSOR_RPP
            data->roiTensorPtrSrc[i].xywhROI.roiWidth = data->srcBatch_width[i];
            data->roiTensorPtrSrc[i].xywhROI.roiHeight = data->srcBatch_height[i];
            data->roiTensorPtrSrc[i].xywhROI.xy.x = 0;
            data->roiTensorPtrSrc[i].xywhROI.xy.y = 0;
        #else
            data->srcDimensions[i].width = data->srcBatch_width[i];
            data->srcDimensions[i].height = data->srcBatch_height[i];
        #endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pSrc, sizeof(data->cl_pSrc)));
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[3], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pDst, sizeof(data->cl_pDst)));
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER, &data->hip_pSrc, sizeof(data->hip_pSrc)));
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[3], VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER, &data->hip_pDst, sizeof(data->hip_pDst)));
        #if TENSOR_RPP
            hipMemcpy(data->d_roiTensorPtrSrc, data->roiTensorPtrSrc, data->nbatchSize * sizeof(RpptROI), hipMemcpyHostToDevice);
        #endif
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &data->pSrc, sizeof(vx_uint8)));
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[3], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &data->pDst, sizeof(vx_uint8)));
    }
    return status;
}

static vx_status VX_CALLBACK validateBrightnessbatchPD(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);
    // Check for input parameters
    vx_parameter input_param;
    vx_image input;
    vx_df_image df_image;
    input_param = vxGetParameterByIndex(node, 0);
    STATUS_ERROR_CHECK(vxQueryParameter(input_param, VX_PARAMETER_ATTRIBUTE_REF, &input, sizeof(vx_image)));
    STATUS_ERROR_CHECK(vxQueryImage(input, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
    if (df_image != VX_DF_IMAGE_U8 && df_image != VX_DF_IMAGE_RGB)
    {
        return ERRMSG(VX_ERROR_INVALID_FORMAT, "validate: BrightnessbatchPD: image: #0 format=%4.4s (must be RGB2 or U008)\n", (char *)&df_image);
    }

    // Check for output parameters
    vx_image output;
    vx_df_image format;
    vx_parameter output_param;
    vx_uint32 height, width;
    output_param = vxGetParameterByIndex(node, 3);
    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_image)));
    STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
    STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[3], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
    vxReleaseImage(&input);
    vxReleaseImage(&output);
    vxReleaseParameter(&output_param);
    vxReleaseParameter(&input_param);
    return status;
}

static vx_status VX_CALLBACK processBrightnessbatchPD(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    #if LOG_LEVEL_2
        #if PROCESS_TIME_LOG
            #if TENSOR_RPP
                std::cerr<<"Complete Process Tensor Time: "<<brightness_complete_process_time<<std::endl;
            #else
                std::cerr<<"Complete Process BatchPD Time: "<<brightness_complete_process_time<<std::endl;
            #endif
        #else
            #if TENSOR_RPP
                std::cerr<<"refreshBrightness Tensor Time: "<<brightness_refresh_time<<std::endl;
                std::cerr<<"Tensor RPP call time: "<<rpp_brigthness_time<<std::endl;
            #else
                std::cerr<<"refreshBrightness batchPD Time: "<<brightness_refresh_time<<std::endl;
                std::cerr<<"BatchPD RPP call time: "<<rpp_brightness_time<<std::endl;
            #endif
        #endif
    #endif
    chrono::high_resolution_clock::time_point process_start_time = chrono::high_resolution_clock::now();

    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    BrightnessbatchPDLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    vx_df_image df_image = VX_DF_IMAGE_VIRT;
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        refreshBrightnessbatchPD(node, parameters, num, data);
        if (df_image == VX_DF_IMAGE_U8)
        {
            rpp_status = rppi_brightness_u8_pln1_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->cl_pDst, data->alpha, data->beta, data->nbatchSize, data->rppHandle);
        }
        else if (df_image == VX_DF_IMAGE_RGB)
        {
            rpp_status = rppi_brightness_u8_pkd3_batchPD_gpu((void *)data->cl_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->cl_pDst, data->alpha, data->beta, data->nbatchSize, data->rppHandle);
        }
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#elif ENABLE_HIP
        chrono::high_resolution_clock::time_point refresh_start_time = chrono::high_resolution_clock::now();
        refreshBrightnessbatchPD(node, parameters, num, data);
        chrono::high_resolution_clock::time_point refresh_end_time = chrono::high_resolution_clock::now();
        chrono::duration<double, std::micro> refresh_time_elapsed = refresh_end_time - refresh_start_time;
        auto refresh_time_dur = static_cast<long long unsigned> (chrono::duration_cast<chrono::microseconds>(refresh_time_elapsed).count());
        brightness_refresh_time +=  refresh_time_dur;

        chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();
        #if TENSOR_RPP
            rpp_status = rppt_brightness_gpu(data->hip_pSrc, data->srcDescPtr, data->hip_pDst, data->dstDescPtr, data->alpha, data->beta, data->d_roiTensorPtrSrc, data->roiType, data->rppHandle);
        #else
            if (df_image == VX_DF_IMAGE_U8)
            {
                rpp_status = rppi_brightness_u8_pln1_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->hip_pDst, data->alpha, data->beta, data->nbatchSize, data->rppHandle);
            }
            else if (df_image == VX_DF_IMAGE_RGB)
            {
                rpp_status = rppi_brightness_u8_pkd3_batchPD_gpu((void *)data->hip_pSrc, data->srcDimensions, data->maxSrcDimensions, (void *)data->hip_pDst, data->alpha, data->beta, data->nbatchSize, data->rppHandle);
            }
        #endif
        chrono::high_resolution_clock::time_point end_time = chrono::high_resolution_clock::now();
        int64_t dur = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
        rpp_brightness_time +=  dur;
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        refreshBrightnessbatchPD(node, parameters, num, data);
        #if TENSOR_RPP
            rpp_status = rppt_brightness_host(data->pSrc, data->srcDescPtr, data->pDst, data->dstDescPtr, data->alpha, data->beta, data->roiTensorPtrSrc, data->roiType, data->rppHandle);
        #else
            if (df_image == VX_DF_IMAGE_U8)
            {
                rpp_status = rppi_brightness_u8_pln1_batchPD_host(data->pSrc, data->srcDimensions, data->maxSrcDimensions, data->pDst, data->alpha, data->beta, data->nbatchSize, data->rppHandle);
            }
            else if (df_image == VX_DF_IMAGE_RGB)
            {
                rpp_status = rppi_brightness_u8_pkd3_batchPD_host(data->pSrc, data->srcDimensions, data->maxSrcDimensions, data->pDst, data->alpha, data->beta, data->nbatchSize, data->rppHandle);
            }
        #endif
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    auto process_time = static_cast<long long unsigned> (chrono::duration_cast<chrono::microseconds>((chrono::high_resolution_clock::now()) - process_start_time).count());
    brightness_complete_process_time +=  process_time;
    auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
    return return_status;
}

static vx_status VX_CALLBACK initializeBrightnessbatchPD(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
    chrono::high_resolution_clock::time_point init_start_time = chrono::high_resolution_clock::now();
    BrightnessbatchPDLocalData *data = new BrightnessbatchPDLocalData;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &data->handle.cmdq, sizeof(data->handle.cmdq)));
#elif ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[6], &data->nbatchSize));
    data->alpha = (vx_float32 *)malloc(sizeof(vx_float32) * data->nbatchSize);
    data->beta = (vx_float32 *)malloc(sizeof(vx_float32) * data->nbatchSize);
    data->srcDimensions = (RppiSize *)malloc(sizeof(RppiSize) * data->nbatchSize);
    data->srcBatch_width = (Rpp32u *)malloc(sizeof(Rpp32u) * data->nbatchSize);
    data->srcBatch_height = (Rpp32u *)malloc(sizeof(Rpp32u) * data->nbatchSize);

    #if TENSOR_RPP
        // Check if it is a RGB or single channel U8 input
        vx_df_image df_image = VX_DF_IMAGE_VIRT;
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
        uint ip_channel = (df_image == VX_DF_IMAGE_RGB) ? 3 : 1;

        // Initializing tensor config parameters.
        data->srcDescPtr = &data->srcDesc;
        data->dstDescPtr = &data->dstDesc;
        data->srcDescPtr->dataType = RpptDataType::U8;
        data->dstDescPtr->dataType = RpptDataType::U8;
        // Set numDims, offset, n/c/h/w values for src/dst
        data->srcDescPtr->numDims = 4;
        data->dstDescPtr->numDims = 4;
        data->srcDescPtr->offsetInBytes = 0;
        data->dstDescPtr->offsetInBytes = 0;
        data->srcDescPtr->n = data->nbatchSize;
        data->srcDescPtr->h = data->maxSrcDimensions.height;
        data->srcDescPtr->w = data->maxSrcDimensions.width;
        data->srcDescPtr->c = ip_channel;
        data->dstDescPtr->n = data->nbatchSize;
        data->dstDescPtr->h = data->maxSrcDimensions.height;
        data->dstDescPtr->w = data->maxSrcDimensions.width;
        data->dstDescPtr->c = ip_channel;
        // Set layout and n/c/h/w strides for src/dst
        if(df_image == VX_DF_IMAGE_U8) // For PLN1 images
        {
            data->srcDescPtr->layout = RpptLayout::NCHW;
            data->dstDescPtr->layout = RpptLayout::NCHW;
            data->srcDescPtr->strides.nStride = ip_channel * data->srcDescPtr->w * data->srcDescPtr->h;
            data->srcDescPtr->strides.cStride = data->srcDescPtr->w * data->srcDescPtr->h;
            data->srcDescPtr->strides.hStride = data->srcDescPtr->w;
            data->srcDescPtr->strides.wStride = 1;
            data->dstDescPtr->strides.nStride = ip_channel * data->dstDescPtr->w * data->dstDescPtr->h;
            data->dstDescPtr->strides.cStride = data->dstDescPtr->w * data->dstDescPtr->h;
            data->dstDescPtr->strides.hStride = data->dstDescPtr->w;
            data->dstDescPtr->strides.wStride = 1;
        }
        else // For RGB (NHWC/NCHW) images
        {
            data->srcDescPtr->layout = RpptLayout::NHWC;
            data->dstDescPtr->layout = RpptLayout::NHWC;
            data->srcDescPtr->strides.nStride = ip_channel * data->srcDescPtr->w * data->srcDescPtr->h;
            data->srcDescPtr->strides.hStride = ip_channel * data->srcDescPtr->w;
            data->srcDescPtr->strides.wStride = ip_channel;
            data->srcDescPtr->strides.cStride = 1;
            data->dstDescPtr->strides.nStride = ip_channel * data->dstDescPtr->w * data->dstDescPtr->h;
            data->dstDescPtr->strides.hStride = ip_channel * data->dstDescPtr->w;
            data->dstDescPtr->strides.wStride = ip_channel;
            data->dstDescPtr->strides.cStride = 1;
        }
        // Initialize ROI tensors, ImagePatch
        data->roiTensorPtrSrc  = (RpptROI *) calloc(data->nbatchSize, sizeof(RpptROI));

        // Set ROI tensors types for src/dst
        data->roiType = RpptRoiType::XYWH;
        #if ENABLE_HIP
            std::cerr<<"Tensor call"<<std::endl;
            hipMalloc(&data->d_roiTensorPtrSrc, data->nbatchSize * sizeof(RpptROI));
        #endif
    #endif

    refreshBrightnessbatchPD(node, parameters, num, data);
#if ENABLE_OPENCL
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppCreateWithStreamAndBatchSize(&data->rppHandle, data->handle.cmdq, data->nbatchSize);
#elif ENABLE_HIP
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppCreateWithStreamAndBatchSize(&data->rppHandle, data->handle.hipstream, data->nbatchSize);
#endif
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
        rppCreateWithBatchSize(&data->rppHandle, data->nbatchSize);

    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    chrono::high_resolution_clock::time_point init_end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, std::micro> init_time_elapsed = init_end_time - init_start_time;
    auto init_time_dur = static_cast<long long unsigned> (chrono::duration_cast<chrono::microseconds>(init_time_elapsed).count());
    brightness_init_time +=  init_time_dur;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeBrightnessbatchPD(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    chrono::high_resolution_clock::time_point uninit_start_time = chrono::high_resolution_clock::now();
    BrightnessbatchPDLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_OPENCL || ENABLE_HIP
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppDestroyGPU(data->rppHandle);
#endif
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
        rppDestroyHost(data->rppHandle);
    free(data->srcBatch_height);
    free(data->srcBatch_width);
    free(data->srcDimensions);
    free(data->alpha);
    free(data->beta);
    #if TENSOR_RPP
        free(data->roiTensorPtrSrc);
        #if ENABLE_HIP
            hipFree(data->d_roiTensorPtrSrc);
        #endif
    #endif

    chrono::high_resolution_clock::time_point uninit_end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, std::micro> uninit_time_elapsed = uninit_end_time - uninit_start_time;
    auto uninit_time_dur = static_cast<long long unsigned> (chrono::duration_cast<chrono::microseconds>(uninit_time_elapsed).count());
    brightness_uninit_time +=  uninit_time_dur;

    delete (data);

    std::cerr<<"\n *******************************************************************************************";
    std::cerr<<"\nComplete Analysis of Brightness augmentation \n";
    #if TENSOR_RPP
        std::cerr<<"Complete Process Tensor Time: "<<brightness_complete_process_time<<std::endl;
        std::cerr<<"RefreshBrightness Tensor Time: "<<brightness_refresh_time<<std::endl;
        std::cerr<<"Tensor RPP call time: "<<rpp_brightness_time<<std::endl;
        std::cerr<<"Uninitialize Brightness Tensor Time: "<<brightness_uninit_time<<std::endl;
        std::cerr<<"Initialize Brightness Tensor Time: "<<brightness_init_time<<std::endl;
    #else
        std::cerr<<"Complete Process BatchPD Time: "<<brightness_complete_process_time<<std::endl;
        std::cerr<<"RefreshBrightness batchPD Time: "<<brightness_refresh_time<<std::endl;
        std::cerr<<"BatchPD RPP call time: "<<rpp_brightness_time<<std::endl;
        std::cerr<<"Uninitialize Brightness batchPD Time: "<<brightness_uninit_time<<std::endl;
        std::cerr<<"Initialize Brightness batchPD Time: "<<brightness_init_time<<std::endl;
    #endif
    std::cerr<<"\n *******************************************************************************************";
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hubrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
)
{
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

// hardcode the affinity to  CPU for OpenCL backend to avoid VerifyGraph failure since there is no codegen callback for amd_rpp nodes
#if ENABLE_OPENCL
    supported_target_affinity = AGO_TARGET_AFFINITY_CPU;
#endif
    return VX_SUCCESS;
}

vx_status BrightnessbatchPD_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.BrightnessbatchPD",
                                       VX_KERNEL_RPP_BRIGHTNESSBATCHPD,
                                       processBrightnessbatchPD,
                                       8,
                                       validateBrightnessbatchPD,
                                       initializeBrightnessbatchPD,
                                       uninitializeBrightnessbatchPD);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_OPENCL || ENABLE_HIP
    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    amd_kernel_query_target_support_f query_target_support_f = query_target_support;

    if (kernel)
    {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS)
    {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }
    return status;
}