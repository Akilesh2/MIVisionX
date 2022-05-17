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

#ifndef MIVISIONX_ROCAL_API_AUGMENTATION_H
#define MIVISIONX_ROCAL_API_AUGMENTATION_H
#include "rocal_api_types.h"

extern "C" RocalTensor ROCAL_API_CALL rocalBrightnessTensor(RocalContext context, RocalTensor input,RocalTensorLayout rocal_tensor_layout,RocalTensorOutputType rocal_tensor_output_type, bool is_output,
                                                   RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL);



/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \param beta
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalBrightness(RocalContext context, RocalTensor input, bool is_output,
                                                   RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param shift
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalBrightnessFixed(RocalContext context, RocalTensor input,
                                                        float alpha, float beta,
                                                        bool is_output);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalGamma(RocalContext context, RocalTensor input,
                                              bool is_output,
                                              RocalFloatParam alpha = NULL);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param alpha
/// \param is_output
/// \return

extern "C" RocalTensor ROCAL_API_CALL rocalGammaTensor(RocalContext context, RocalTensor input,RocalTensorLayout rocal_tensor_layout,RocalTensorOutputType rocal_tensor_output_type, bool is_output,
                                                   RocalFloatParam alpha = NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalExposureTensor(RocalContext context, RocalTensor input,RocalTensorLayout rocal_tensor_layout,RocalTensorOutputType rocal_tensor_output_type, bool is_output,
                                                   RocalFloatParam alpha = NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalColorTwistTensor(RocalContext context, RocalTensor input,RocalTensorLayout rocal_tensor_layout,RocalTensorOutputType rocal_tensor_output_type, bool is_output,
                                                   RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL, RocalFloatParam hue = NULL, RocalFloatParam sat = NULL);

extern "C" RocalTensor ROCAL_API_CALL rocalCropTensor(RocalContext context, RocalTensor input,
                                                                  RocalTensorLayout rocal_tensor_layout,
                                                                  RocalTensorOutputType rocal_tensor_output_type,
                                                                  unsigned crop_depth,
                                                                  unsigned crop_height,
                                                                  unsigned crop_width,
                                                                  float start_x,
                                                                  float start_y,
                                                                  float start_z,
                                                                  bool is_output);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \return

extern "C" RocalTensor ROCAL_API_CALL rocalGammaFixed(RocalContext context, RocalTensor input, float alpha, bool is_output);

extern "C" RocalTensor ROCAL_API_CALL rocalCopyTensor(RocalContext context, RocalTensor input, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \return

extern "C" RocalTensor ROCAL_API_CALL rocalNopTensor(RocalContext context, RocalTensor input, bool is_output);


extern "C" RocalTensor ROCAL_API_CALL rocalCropMirrorNormalizeTensor(RocalContext context, RocalTensor input,
                                                                  RocalTensorLayout rocal_tensor_layout,
                                                                  RocalTensorOutputType rocal_tensor_output_type,
                                                                  unsigned crop_depth,
                                                                  unsigned crop_height,
                                                                  unsigned crop_width,
                                                                  float start_x,
                                                                  float start_y,
                                                                  float start_z,
                                                                  std::vector<float> &mean,
                                                                  std::vector<float> &std_dev,
                                                                  bool is_output,
                                                                  RocalIntParam mirror = NULL);


#endif //MIVISIONX_ROCAL_API_AUGMENTATION_H