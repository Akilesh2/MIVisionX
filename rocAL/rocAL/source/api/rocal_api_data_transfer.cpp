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

#include "commons.h"
#include "context.h"
#include "rocal_api.h"
#if !ENABLE_HIP
#include "CL/cl.h"
#endif

RocalStatus ROCAL_API_CALL
rocalCopyToOutput(
        RocalContext p_context,
        void* out_ptr,
        size_t out_size)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        context->master_graph->copy_output(out_ptr, out_size);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

RocalStatus ROCAL_API_CALL
rocalCopyToOutput(
        RocalContext p_context,
        unsigned char * out_ptr,
        size_t out_size)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        context->master_graph->copy_output(out_ptr);
        // std::cerr<<"\n commented  context->master_graph->copy_output(out_ptr";
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

RocalStatus ROCAL_API_CALL
rocalCopyToTensorOutput(
        RocalContext p_context,
        void * out_ptr,
        size_t out_size)
{
    std::cerr<<"rocalCopyToTensorOutput";
    auto context = static_cast<Context*>(p_context);
    try
    {
        context->master_graph->copy_output(out_ptr);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}
