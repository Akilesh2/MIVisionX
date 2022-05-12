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

#include <vx_ext_rpp.h>
#include "node_brightness.h"
#include "exception.h"

BrightnessTensorNode::BrightnessTensorNode(const std::vector<rocALTensor *> &inputs,const std::vector<rocALTensor *> &outputs) :
        TensorNode(inputs, outputs),
        _alpha(ALPHA_RANGE[0], ALPHA_RANGE[1]),
        _beta (BETA_RANGE[0], BETA_RANGE[1])
{
}

void BrightnessTensorNode::create_node()
{
    if(_node)
        return;

    _alpha.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _beta.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);

    if(_inputs[0]->info().layout() == RocalTensorlayout::NCHW)
        _layout = 1;
    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);

    _node = vxExtrppNode_Brightness(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _alpha.default_array(), _beta.default_array(), layout, roi_type, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the brightness_batch (vxExtrppNode_Brightness) node failed: "+ TOSTR(status))
}

void BrightnessTensorNode::init( float alpha, float beta)
{
    _alpha.set_param(alpha);
    _beta.set_param(beta);
    _layout = _roi_type = 0;
}

void BrightnessTensorNode::init( FloatParam* alpha, FloatParam* beta)
{
    _alpha.set_param(core(alpha));
    _beta.set_param(core(beta));
    _layout = _roi_type = 0;
}


void BrightnessTensorNode::update_node()
{
    _alpha.update_array();
    _beta.update_array();
}


//***********************************Brightness***********************************************************
BrightnessNode::BrightnessNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) :
        Node(inputs, outputs),
        _alpha(ALPHA_RANGE[0], ALPHA_RANGE[1]),
        _beta (BETA_RANGE[0], BETA_RANGE[1])
{
}

void BrightnessNode::create_node()
{
    if(_node)
        return;

    _alpha.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _beta.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);

    // _node = vxExtrppNode_Brightness(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _alpha.default_array(), _beta.default_array(), _batch_size);/*A temporary fix for time being*/

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the brightness_batch (vxExtrppNode_BrightnessbatchPD) node failed: "+ TOSTR(status))
}

void BrightnessNode::init( float alpha, float beta)
{
    _alpha.set_param(alpha);
    _beta.set_param(beta);
}

void BrightnessNode::init( FloatParam* alpha, FloatParam* beta)
{
    _alpha.set_param(core(alpha));
    _beta.set_param(core(beta));
}


void BrightnessNode::update_node()
{
    _alpha.update_array();
    _beta.update_array();
}


