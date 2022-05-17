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
#include <graph.h>
#include "node_crop.h"
#include "exception.h"

CropTensorNode::CropTensorNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs) :
        TensorNode(inputs, outputs)
{
    _crop_param = std::make_shared<RocalCropParam>(_batch_size);
}

void CropTensorNode::create_node()
{
    if(_node)
        return;

    if(_crop_param->crop_h == 0 || _crop_param->crop_w == 0)
        THROW("Uninitialized destination dimension - Invalid Crop Sizes")
    _crop_param->create_array(_graph);
   
    unsigned int chnShift = 0;
    vx_scalar  chnToggle = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&chnShift);
    bool packed;
    if(_inputs[0]->info().color_format() != RocalColorFormat::RGB_PLANAR)
    {
        packed = true;
    }
    vx_scalar is_packed = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_BOOL,&packed);
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);
    _node = vxExtrppNode_Crop(_graph->get(), _inputs[0]->handle(),
                                                   _src_tensor_roi,_outputs[0]->handle(),_src_tensor_roi,_crop_param->cropw_arr, _crop_param->croph_arr, _crop_param->x1_arr, _crop_param->y1_arr,
                                                   is_packed, chnToggle,layout, roi_type, _batch_size);
    vx_status status = VX_SUCCESS;

    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop tensor (vxExtrppNode_CropbatchPD    ) failed: "+TOSTR(status))

}


void CropTensorNode::update_node()
{
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi());

    _crop_param->update_array();
    std::vector<uint32_t> crop_h_dims, crop_w_dims;
    _crop_param->get_crop_dimensions(crop_w_dims, crop_h_dims);
    _outputs[0]->update_tensor_roi(crop_w_dims, crop_h_dims);

    
}

void CropTensorNode::init(int crop_h, int crop_w, float start_x, float start_y)
{
    _crop_param->crop_h = crop_h;
    _crop_param->crop_w = crop_w;

}



