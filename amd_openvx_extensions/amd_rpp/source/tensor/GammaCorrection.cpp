/*
Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
#define NUM_OF_DIMS 4

struct GammaCorrectionLocalData
{
    RPPCommonHandle handle;
    rppHandle_t rppHandle;
    Rpp32u device_type;
    Rpp32u nbatchSize;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    vx_float32 *gamma;
    vx_size channels;
    RpptDescPtr src_desc_ptr;
    // RpptDescPtr dst_desc_ptr;

    RpptDesc srcDesc;
    RpptDesc dstDesc;

    RpptROI *roi_tensor_Ptr;
    RpptRoiType roiType;
    size_t in_tensor_dims[NUM_OF_DIMS]; // will have NHWC info
    size_t out_tensor_dims[NUM_OF_DIMS];

    vx_enum in_tensor_type ;
    vx_enum out_tensor_type; // will have NHWC info
#if ENABLE_OPENCL
    cl_mem cl_pSrc;
    cl_mem cl_pDst;
#elif ENABLE_HIP
    void *hip_pSrc;
    void *hip_pDst;
#endif
};

static vx_status VX_CALLBACK refreshGammaCorrection(vx_node node, const vx_reference *parameters, vx_uint32 num, GammaCorrectionLocalData *data)
{

    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[1], 0, data->nbatchSize *4, sizeof(unsigned),data->roi_tensor_Ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[3], 0, data->nbatchSize, sizeof(vx_float32), data->gamma, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->cl_pSrc, sizeof(data->cl_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &data->cl_pDst, sizeof(data->cl_pDst)));
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->hip_pSrc, sizeof(data->hip_pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->hip_pDst, sizeof(data->hip_pDst)));
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        // STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
        // STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_uint8)));
        if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8 && data->out_tensor_type == vx_type_e::VX_TYPE_UINT8)
        {
            std::cerr<<"\n ************************************************* U8 to U8 Gonna copy tensor source buffer*******************************************";

            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_uint8)));
        }
        else if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32 && data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            std::cerr<<"*******************FLOAT32*******************";
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float32)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
        }
        // vx_float16 is not supported. Have to disable it once it is done.
        // else if (in_tensor_type == vx_type_e::VX_TYPE_FLOAT16 && out_tensor_type == vx_type_e::VX_TYPE_FLOAT16)
        // {
        //     
            // STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_float16)));
        //     STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float16)));
        // }
        else if (data->in_tensor_type == vx_type_e::VX_TYPE_INT8 && data->out_tensor_type == vx_type_e::VX_TYPE_INT8)
        {

            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_int8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_int8)));
        }
        else if (data->in_tensor_type == vx_type_e::VX_TYPE_UINT8 && data->out_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
        {
            std::cerr<<"\n *************************************************U8 to F32 Gonna copy tensor source buffer*******************************************";

            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(vx_uint8)));
            STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(vx_float32)));
        }
    }
    return status;
}

static vx_status VX_CALLBACK validateGammaCorrection(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    std::cerr<<"GAMMA";
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #4 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #5 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);

    // Check for output parameters
    vx_tensor output;
    vx_parameter output_param;
    size_t num_tensor_dims;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[4];
    vx_enum tensor_type;
    output_param = vxGetParameterByIndex(node, 2);
    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_tensor)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    vxReleaseTensor(&output);
    vxReleaseParameter(&output_param);
    return status;
}

static vx_status VX_CALLBACK processGammaCorrection(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    GammaCorrectionLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        refreshGammaCorrection(node, parameters, num, data);
        rpp_status = rppt_gamma_correction_gpu((void *)data->cl_pSrc, data->src_desc_ptr, (void *)data->cl_pDst, data->src_desc_ptr,  data->gamma, data->roi_tensor_Ptr, data->roiType, data->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#elif ENABLE_HIP
        refreshGammaCorrection(node, parameters, num, data);
        rpp_status = rppt_gamma_correction_gpu((void *)data->hip_pSrc, data->src_desc_ptr, (void *)data->hip_pDst, data->src_desc_ptr,  data->gamma, data->roi_tensor_Ptr, data->roiType, data->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        refreshGammaCorrection(node, parameters, num, data);
        for(int i = 0; i < data->nbatchSize; i++)
        {

            std::cerr<<"\nbbox values :: "<<data->roi_tensor_Ptr[i].xywhROI.xy.x<<" "<<data->roi_tensor_Ptr[i].xywhROI.xy.y<<" "<<data->roi_tensor_Ptr[i].xywhROI.roiWidth<<" "<<data->roi_tensor_Ptr[i].xywhROI.roiHeight;
        }
        std::cerr<<"\nDatatype  "<<data->in_tensor_type<<" \n"<<"gamma";
        std::cerr<<"before gamma rpp call\n";
        float *temp = ((float*)calloc( 100,sizeof(float) ));

        for (int i=0;i< 100;i++)
                {
                    temp[i]=*((float *)(data->pSrc) + i);
                    // *((float *)(data->pSrc) + i)=*((float *)(data->) + i)*255;
                    std::cout<<temp[i]<<" ";

                }
            
        rpp_status = rppt_gamma_correction_host(data->pSrc, data->src_desc_ptr, data->pDst, data->src_desc_ptr, data->gamma, data->roi_tensor_Ptr, data->roiType, data->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
        std::cerr<<"\n back from RPP";
                // float *temp = ((float*)calloc( 100,sizeof(float) ));
        std::cerr<<"After gamma rpp call \n";
        for (int i=0;i< 100;i++)
                {
                    temp[i]=(*((float *)(data->pDst) + i));
                    // *((float *)(data->pDst) + i)=*((float *)(data->pDst) + i)*255;
                    std::cout<<temp[i]<<" ";

                }
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeGammaCorrection(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    GammaCorrectionLocalData *data = new GammaCorrectionLocalData;
    unsigned layout,roiType;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &data->handle.cmdq, sizeof(data->handle.cmdq)));
#elif ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->handle.hipstream, sizeof(data->handle.hipstream)));
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[6], &data->nbatchSize));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[4], &layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    std::cerr<<"\n layout "<<layout;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &roiType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(roiType == 1)
        data->roiType = RpptRoiType::XYWH;
    else
        data->roiType = RpptRoiType::LTRB;
    data->src_desc_ptr = &data->srcDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->src_desc_ptr->numDims, sizeof(data->src_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->in_tensor_dims, sizeof(vx_size) * data->src_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0],VX_TENSOR_DATA_TYPE, &data->in_tensor_type, sizeof(data->in_tensor_type)));

    data->out_tensor_type = data->in_tensor_type; //for gamma_correction augmentation RPP supports only same datatype 
    if(data->in_tensor_type == vx_type_e::VX_TYPE_UINT8)
    {
        std::cerr<<"datatype check UINT8";

        data->src_desc_ptr->dataType = RpptDataType::U8;
    }
    else if (data->in_tensor_type == vx_type_e::VX_TYPE_FLOAT32)
    {
        std::cerr<<"datatype check FLOAT32";

        data->src_desc_ptr->dataType = RpptDataType::F32;
    }
    // else if (data->src_desc_ptr->dataType == vx_type_e::VX_TYPE_FLOAT16)
    //     data->src_desc_ptr->dataType = RpptDataType::F16;
    else if (data->in_tensor_type == vx_type_e::VX_TYPE_INT8)
    {
        std::cerr<<"datatype check INT8";

        data->src_desc_ptr->dataType = RpptDataType::I8;
    }


    // if(data->src_desc_ptr->dataType == vx_type_e::VX_TYPE_UINT8)
    //     data->src_desc_ptr->dataType = RpptDataType::U8;
     data->src_desc_ptr->offsetInBytes = 0;


    
    if(layout == 0) // NHWC
    {
        data->src_desc_ptr->n = data->in_tensor_dims[0];
        data->src_desc_ptr->h = data->in_tensor_dims[1];
        data->src_desc_ptr->w = data->in_tensor_dims[2];
        data->src_desc_ptr->c = data->in_tensor_dims[3];
        std::cerr<<"\n n h w c "<<data->src_desc_ptr->n<<" "<<data->src_desc_ptr->h<<" "<<data->src_desc_ptr->w<<" "<<data->src_desc_ptr->c;
        data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->src_desc_ptr->strides.hStride = data->src_desc_ptr->c * data->src_desc_ptr->w;
        data->src_desc_ptr->strides.wStride = data->src_desc_ptr->c;
        data->src_desc_ptr->strides.cStride = 1;
        data->src_desc_ptr->layout = RpptLayout::NHWC;
        std::cerr<<"\n Setting layout "<<data->src_desc_ptr->layout;
        std::cerr<<"\n Setting data type "<<data->src_desc_ptr->dataType;

    }
    else // NCHW
    {
        data->src_desc_ptr->n = data->in_tensor_dims[0];
        data->src_desc_ptr->h = data->in_tensor_dims[2];
        data->src_desc_ptr->w = data->in_tensor_dims[3];
        data->src_desc_ptr->c = data->in_tensor_dims[1];
        
        data->src_desc_ptr->strides.nStride = data->src_desc_ptr->c * data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->src_desc_ptr->strides.cStride = data->src_desc_ptr->w * data->src_desc_ptr->h;
        data->src_desc_ptr->strides.hStride = data->src_desc_ptr->w;
        data->src_desc_ptr->strides.wStride = 1;
        data->src_desc_ptr->layout = RpptLayout::NCHW;

    }
        
    // data->dst_desc_ptr = data->src_desc_ptr;

    data->roi_tensor_Ptr = (RpptROI *) calloc(data->nbatchSize, sizeof(RpptROI));
    data->gamma = (vx_float32 *)malloc(sizeof(vx_float32) * data->nbatchSize);
    // data->src_desc_ptr->dataType=RpptDataType::F32;   
    // data->dst_desc_ptr->dataType=RpptDataType::F32;
    refreshGammaCorrection(node, parameters, num, data);
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
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeGammaCorrection(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    GammaCorrectionLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_OPENCL || ENABLE_HIP
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
        rppDestroyGPU(data->rppHandle);
#endif
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
        rppDestroyHost(data->rppHandle);
    free(data->roi_tensor_Ptr);
    free(data->gamma);
    delete (data);
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

vx_status GammaCorrection_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.GammaCorrection",
                                       VX_KERNEL_RPP_GAMMACORRECTION,
                                       processGammaCorrection,
                                       8,
                                       validateGammaCorrection,
                                       initializeGammaCorrection,
                                       uninitializeGammaCorrection);
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
       
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
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