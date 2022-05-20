#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"
#include "parameter_vx.h"
#include "rocal_api_types.h"
//final
class ResizeTensorNode : public TensorNode
{
public:
    ResizeTensorNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs);
    ResizeTensorNode() = delete;
    void init(int interpolation_type);

    unsigned int get_dst_width() { return 100; }
    unsigned int get_dst_height() { return 100; }
    vx_array get_src_width() { return _src_roi_width; }
    vx_array get_src_height() { return _src_roi_height; }
protected:
    void create_node() override ;
    void update_node() override;
private:
    vx_array  _dst_roi_width , _dst_roi_height,_src_roi_width, _src_roi_height;
    unsigned _layout, _roi_type;

    int _interpolation_type;

};







// class ResizeTensorNode : public TensorNode
// {
// public:
//     ResizeTensorNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs);
//     ResizeTensorNode() = delete;
//     void init(int crop_h, int crop_w, int interpolation_type);
//     std::shared_ptr<RocalCropParam> return_crop_param() { return _crop_param; }
//     vx_array get_src_width() { return _src_roi_width; }
//     vx_array get_src_height() { return _src_roi_height; }
// protected:
//     void create_node() override ;
//     void update_node() override;
// private:
//     std::shared_ptr<RocalCropParam> _crop_param;
//     vx_array _src_roi_width,_src_roi_height;
//     int _interpolation_type;
//     unsigned _layout, _roi_type;
// };

