/*
Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

/************************************************************************************************************
input parameter validator.
param [in] node The handle to the node.
param [in] index The index of the parameter to validate.
*************************************************************************************************************/
vx_status VX_CALLBACK CV_adaptiveThreshold_InputValidator(vx_node node, vx_uint32 index)
{
    vx_status status = VX_SUCCESS;
    vx_parameter param = vxGetParameterByIndex(node, index);

    if (index == 0)
    {
        vx_image image;
        vx_df_image df_image = VX_DF_IMAGE_VIRT;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)));
        STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
        if (df_image != VX_DF_IMAGE_U8)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseImage(&image);
    }

    else if (index == 1)
    {
        vx_image image;
        vx_df_image df_image = VX_DF_IMAGE_VIRT;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)));
        STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
        if (df_image != VX_DF_IMAGE_U8)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseImage(&image);
    }

    else if (index == 2)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_float32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 0 || type != VX_TYPE_FLOAT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    else if (index == 3)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_int32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 0 || type != VX_TYPE_INT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    else if (index == 4)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_int32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 0 || type != VX_TYPE_INT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    else if (index == 5)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_int32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 3 || (value % 2 == 0) || type != VX_TYPE_INT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    else if (index == 6)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_float32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (type != VX_TYPE_FLOAT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    vxReleaseParameter(&param);
    return status;
}

/************************************************************************************************************
output parameter validator.
*************************************************************************************************************/
vx_status VX_CALLBACK CV_adaptiveThreshold_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
    vx_status status = VX_SUCCESS;
    if (index == 1)
    {
        vx_parameter output_param = vxGetParameterByIndex(node, 1);
        vx_image output;
        vx_uint32 width = 0, height = 0;
        vx_df_image format = VX_DF_IMAGE_VIRT;

        STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_image)));
        STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
        STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
        STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));

        if (format != VX_DF_IMAGE_U8)
        {
            status = VX_ERROR_INVALID_VALUE;
        }

        STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
        STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
        STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
        vxReleaseImage(&output);
        vxReleaseParameter(&output_param);
    }
    return status;
}

/************************************************************************************************************
Execution Kernel
*************************************************************************************************************/
vx_status VX_CALLBACK CV_adaptiveThreshold_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_status status = VX_SUCCESS;

    vx_image image_in = (vx_image)parameters[0];
    vx_image image_out = (vx_image)parameters[1];
    vx_scalar scalar = (vx_scalar)parameters[2];
    vx_scalar ADAPTIVE_METHOD = (vx_scalar)parameters[3];
    vx_scalar THRESHOLD_TYPE = (vx_scalar)parameters[4];
    vx_scalar BLOCKSIZE = (vx_scalar)parameters[5];
    vx_scalar C = (vx_scalar)parameters[6];

    Mat *mat, bl;

    int adaptiveMethod, thresholdType, blockSize;
    float maxValue, c;
    vx_int32 value = 0;
    vx_float32 value_f = 0;

    //Extracting Values from the Scalar
    STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value_f));
    maxValue = value_f;
    STATUS_ERROR_CHECK(vxReadScalarValue(ADAPTIVE_METHOD, &value));
    adaptiveMethod = value;
    STATUS_ERROR_CHECK(vxReadScalarValue(THRESHOLD_TYPE, &value));
    thresholdType = value;
    STATUS_ERROR_CHECK(vxReadScalarValue(BLOCKSIZE, &value));
    blockSize = value;
    STATUS_ERROR_CHECK(vxReadScalarValue(C, &value_f));
    c = value_f;

    //Converting VX Image to OpenCV Mat
    STATUS_ERROR_CHECK(match_vx_image_parameters(image_in, image_out));
    STATUS_ERROR_CHECK(VX_to_CV_Image(&mat, image_in));

    //Compute using OpenCV
    cv::adaptiveThreshold(*mat, bl, maxValue, adaptiveMethod, thresholdType, blockSize, c);

    //Converting OpenCV Mat into VX Image
    STATUS_ERROR_CHECK(CV_to_VX_Image(image_out, &bl));

    return status;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status CV_adaptiveThreshold_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    vx_kernel kernel = vxAddKernel(context,
                                   "org.opencv.adaptivethreshold",
                                   VX_KERNEL_OPENCV_ADAPTIVETHRESHOLD,
                                   CV_adaptiveThreshold_Kernel,
                                   7,
                                   CV_adaptiveThreshold_InputValidator,
                                   CV_adaptiveThreshold_OutputValidator,
                                   nullptr,
                                   nullptr);

    if (kernel)
    {
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
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
