#include <vx_ext_rpp.h>
#include "node_color_cast.h"
#include "exception.h"

ColorCastTensorNode::ColorCastTensorNode(const std::vector<rocALTensor *> &inputs,const std::vector<rocALTensor *> &outputs) :
        TensorNode(inputs, outputs),
        _red(RED_RANGE[0], RED_RANGE[1]),
        _green (GREEN_RANGE[0], GREEN_RANGE[1]),
        _blue(BLUE_RANGE[0], BLUE_RANGE[1]),
        _alpha(ALPHA_RANGE[0], ALPHA_RANGE[1])


{
}

void ColorCastTensorNode::create_node()
{
    if(_node)
        return;

    _red.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _green.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _blue.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _alpha.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);

    if(_inputs[0]->info().layout() == RocalTensorlayout::NCHW)
        _layout = 1;
    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);

    _node = vxExtrppNode_ColorCast(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _red.default_array(), _green.default_array(), _blue.default_array(), _alpha.default_array(), layout, roi_type, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the colorcast_batch (vxExtrppNode_ColotCast) node failed: "+ TOSTR(status))
}

void ColorCastTensorNode::init( float red, float green, float blue , float alpha,int layout)
{
    _layout=layout;
    _red.set_param(red);
    _green.set_param(green);
    _blue.set_param(blue);
    _alpha.set_param(alpha);
    // _layout = _roi_type = 0;
}

void ColorCastTensorNode::init( FloatParam* red, FloatParam* green, FloatParam* blue, FloatParam* alpha, int layout)
{
    _layout=layout;
    _red.set_param(core(red));
    _green.set_param(core(green));
    _blue.set_param(core(blue));
    _alpha.set_param(core(alpha));
    // _layout = _roi_type = 0;
}


void ColorCastTensorNode::update_node()
{
    _red.update_array();
    _green.update_array();
    _blue.update_array();
    _alpha.update_array();
}
