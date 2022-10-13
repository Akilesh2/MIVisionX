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

/*
 * dtypes.h
 *
 *  Created on: Jan 29, 2019
 *      Author: root
 */

#pragma once
#include <vector>
#include "exception.h"
#include "log.h"

/*! \brief Tensor layouts
 *
 * currently supported by Rocal SDK as input/output
 */
enum class RocalTensorlayout
{
    NHWC = 0,
    NCHW,
    NFHWC,
    NFCHW,
    NONE
};

/*! \brief Tensor data type
 *
 */
enum class RocalTensorDataType
{
    FP32 = 0,
    FP16,
    UINT8,
    UINT32,
    INT32
};

enum class RocalAffinity
{
    GPU = 0,
    CPU
};

/*! \brief Color formats currently supported by Rocal SDK as input/output
 *
 */
enum class RocalColorFormat
{
    RGB24 = 0,
    BGR24,
    U8,
    RGB_PLANAR,
};

/*! \brief Memory type, host or device
 *
 *  Currently supports HOST and OCL, will support HIP in future
 */
enum class RocalMemType
{
    HOST = 0,
    OCL,
    HIP
};

/*! \brief Decoder mode for Video decoding
 *
 *  Currently supports Software decoding, will support Hardware decoding in future
 */
enum class DecodeMode
{
    HW_VAAPI = 0,
    CPU = 1
};

/*! \brief Tensor ROI type
 *
 * currently supports following formats
 */
enum class RocalROIType
{
    XYWH = 0,
    LTRB
};

typedef struct
{
    unsigned x1, y1, x2, y2;
} RocalROI;

struct Timing
{
    // The following timings are accumulated timing not just the most recent activity
    long long unsigned image_read_time= 0;
    long long unsigned image_decode_time= 0;
    long long unsigned to_device_xfer_time= 0;
    long long unsigned from_device_xfer_time= 0;
    long long unsigned copy_to_output = 0;
    long long unsigned image_process_time= 0;
    long long unsigned bb_process_time= 0;
    long long unsigned mask_process_time= 0;
    long long unsigned label_load_time= 0;
    long long unsigned bb_load_time= 0;
    long long unsigned mask_load_time = 0;
    long long unsigned shuffle_time = 0;
    long long unsigned video_read_time= 0;
    long long unsigned video_decode_time= 0;
    long long unsigned video_process_time= 0;
};