#include <vx_ext_rpp.h>
#include "node_color_twist.h"
#include "exception.h"

ColorTwistTensorNode::ColorTwistTensorNode(const std::vector<rocALTensor *> &inputs,const std::vector<rocALTensor *> &outputs) :
        TensorNode(inputs, outputs),
        _alpha(ALPHA_RANGE[0], ALPHA_RANGE[1]),
        _beta (BETA_RANGE[0], BETA_RANGE[1]),
        _hue(HUE_RANGE[0], HUE_RANGE[1]),
        _sat(SAT_RANGE[0], SAT_RANGE[1])


{
}

void ColorTwistTensorNode::create_node()
{
    if(_node)
        return;

    _alpha.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _beta.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _hue.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _sat.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);

    if(_inputs[0]->info().layout() == RocalTensorlayout::NCHW)
        _layout = 1;
    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);

    _node = vxExtrppNode_ColorTwist(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _alpha.default_array(), _beta.default_array(), _hue.default_array(), _sat.default_array(), layout, roi_type, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the colortwist_batch (vxExtrppNode_ColotTwsit) node failed: "+ TOSTR(status))
}

void ColorTwistTensorNode::init( float alpha, float beta, float hue , float sat)
{
    _alpha.set_param(alpha);
    _beta.set_param(beta);
    _hue.set_param(hue);
    _sat.set_param(sat);
    _layout = _roi_type = 0;
}

void ColorTwistTensorNode::init( FloatParam* alpha, FloatParam* beta, FloatParam* hue, FloatParam* sat)
{
    _alpha.set_param(core(alpha));
    _beta.set_param(core(beta));
    _hue.set_param(core(hue));
    _sat.set_param(core(sat));
    _layout = _roi_type = 0;
}


void ColorTwistTensorNode::update_node()
{
    _alpha.update_array();
    _beta.update_array();
    _hue.update_array();
    _sat.update_array();
}
