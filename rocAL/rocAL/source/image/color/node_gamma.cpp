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
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "node_gamma.h"
#include "exception.h"

GammaTensorNode::GammaTensorNode(const std::vector<rocALTensor *> &inputs,const std::vector<rocALTensor *> &outputs) :
        TensorNode(inputs, outputs),
        _shift(SHIFT_RANGE[0], SHIFT_RANGE[1])
{
}

void GammaTensorNode::create_node()
{
    if(_node)
        return;

    _shift.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);

    if(_inputs[0]->info().layout() == RocalTensorlayout::NCHW)
        _layout = 1;
    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);

    _node = vxExtrppNode_GammaCorrection(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _shift.default_array(), layout, roi_type, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the brightness_batch (vxExtrppNode_BrightnessbatchPD) node failed: "+ TOSTR(status))
}

void GammaTensorNode::init( float shift)
{
    _shift.set_param(shift);
    _layout = _roi_type = 0;
}

void GammaTensorNode::init( FloatParam* shift)
{
    _shift.set_param(core(shift));
    _layout = _roi_type = 0;
}


void GammaTensorNode::update_node()
{
    _shift.update_array();
}
//*****************************GAMMA*********************************************//
GammaNode::GammaNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) :
        Node(inputs, outputs),
        _shift(SHIFT_RANGE[0], SHIFT_RANGE[1])
{
}

void GammaNode::create_node()
{
    if(_node)
        return;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _shift.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    // _node = vxExtrppNode_GammaCorrection(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _shift.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the gamma (vxExtrppNode_GammaCorrectionbatchPD) node failed: "+ TOSTR(status))

}

void GammaNode::init(float shfit)
{
    _shift.set_param(shfit);
}

void GammaNode::init(FloatParam* shfit)
{
    _shift.set_param(core(shfit));
}

void GammaNode::update_node()
{
     _shift.update_array();
}