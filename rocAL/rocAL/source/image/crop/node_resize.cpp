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

// #include <vx_ext_rpp.h>
// #include <graph.h>
// #include "node_resize.h"
// #include "exception.h"

// ResizeTensorNode::ResizeTensorNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs) :
//         TensorNode(inputs, outputs)
// {
//     _crop_param = std::make_shared<RocalCropParam>(_batch_size);
// }

// void ResizeTensorNode::create_node()
// {
//     if(_node)
//         return;

//     // if(_crop_param->crop_h == 0 || _crop_param->crop_w == 0)
//     //     THROW("Uninitialized destination dimension - Invalid Crop Sizes")
//     // _crop_param->create_array(_graph);
//     std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().width());
//     std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().height_single());

//     _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
//     _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
//     unsigned int chnShift = 0;
//     vx_scalar  chnToggle = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&chnShift);
//     bool packed;
//     if(_inputs[0]->info().color_format() != RocalColorFormat::RGB_PLANAR)
//     {
//         packed = true;
//     }
//     vx_scalar is_packed = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_BOOL,&packed);
//     vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
//     vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);
//     vx_scalar interpolation_type =vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_interpolation_type);

//     _node = vxExtrppNode_Resize(_graph->get(), _inputs[0]->handle(),
//                                                    _src_tensor_roi,_outputs[0]->handle(),_src_tensor_roi,_crop_param->cropw_arr, _crop_param->croph_arr, interpolation_type,
//                                                    is_packed, chnToggle,layout, roi_type, _batch_size);
//     vx_status status = VX_SUCCESS;

//     if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
//         THROW("Error adding the crop tensor (vxExtrppNode_CropbatchPD    ) failed: "+TOSTR(status))

// }


// void ResizeTensorNode::update_node()
// {
//     // _crop_param->set_image_dimensions(_inputs[0]->info().get_roi());

//     // _crop_param->update_array();
//     // std::vector<uint32_t> crop_h_dims, crop_w_dims;
//     // _crop_param->get_crop_dimensions(crop_w_dims, crop_h_dims);
//     // _outputs[0]->update_tensor_roi(crop_w_dims, crop_h_dims);

    
// }

// void ResizeTensorNode::init(int crop_h, int crop_w, int interpolation_type)
// {
//     _crop_param->crop_h = crop_h;
//     _crop_param->crop_w = crop_w;
//     _interpolation_type= interpolation_type;
// }

//*************************************tensor***************************************

//final

#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_resize.h"
#include "exception.h"
ResizeTensorNode::ResizeTensorNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs) :
        TensorNode(inputs, outputs)
{
}
    
void ResizeTensorNode::create_node()
{
    if(_node)
        return;
    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().get_width());
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().get_height());

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    std::cerr<<"dst_roi_height "<<dst_roi_height[0]<<"  "<<dst_roi_width[0];
     if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status));
    bool packed;
    vx_scalar interpolation = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_interpolation_type);

    unsigned int chnShift = 0;
    vx_scalar  chnToggle = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&chnShift);
    vx_scalar is_packed = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_BOOL,&packed);

    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);

   _node = vxExtrppNode_Resize(_graph->get(), _inputs[0]->handle(),
                                                   _src_tensor_roi,_outputs[0]->handle(),_src_tensor_roi,_dst_roi_width,_dst_roi_height,interpolation,
                                                 is_packed, chnToggle,layout, roi_type, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize (vxExtrppNode_ResizebatchPD) node failed: "+ TOSTR(status))

}
void ResizeTensorNode::update_node()
{
  
}
void ResizeTensorNode::init(int interpolation_type, int layout)
{
  _interpolation_type=interpolation_type;
  _layout=layout;
  
}





//*******************************************resize************************************
/*
#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_resize.h"
#include "exception.h"


ResizeNode::ResizeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs)
{
}

void ResizeNode::create_node()
{
    if(_node)
        return;

    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().width());
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().height_single());

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
     if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

   _node = vxExtrppNode_ResizebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize (vxExtrppNode_ResizebatchPD) node failed: "+ TOSTR(status))

}

void ResizeNode::update_node()
{
  
}












*/