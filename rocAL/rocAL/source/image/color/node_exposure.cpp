#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "node_exposure.h"
#include "exception.h"

ExposureTensorNode::ExposureTensorNode(const std::vector<rocALTensor *> &inputs,const std::vector<rocALTensor *> &outputs) :
        TensorNode(inputs, outputs),
        _shift(SHIFT_RANGE[0], SHIFT_RANGE[1])
{
}

void ExposureTensorNode::create_node()
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

    _node = vxExtrppNode_Exposure(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _shift.default_array(), layout, roi_type, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the Exposure_batch (vxExtrppNode_Exposure) node failed: "+ TOSTR(status))
}

void ExposureTensorNode::init( float shift)
{
    _shift.set_param(shift);
    _layout = _roi_type = 0;
}

void ExposureTensorNode::init( FloatParam* shift)
{
    _shift.set_param(core(shift));
    _layout = _roi_type = 0;
}


void ExposureTensorNode::update_node()
{
    _shift.update_array();
}