
#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"
#include "parameter_vx.h"
#include "rocal_api_types.h"

class CropTensorNode : public TensorNode
{
public:
    CropTensorNode(const std::vector<rocALTensor *> &inputs, const std::vector<rocALTensor *> &outputs);
    CropTensorNode() = delete;
    void init(int crop_h, int crop_w, float start_x, float start_y,int layout);
    std::shared_ptr<RocalCropParam> return_crop_param() { return _crop_param; }
    vx_array get_src_width() { return _src_roi_width; }
    vx_array get_src_height() { return _src_roi_height; }
protected:
    void create_node() override ;
    void update_node() override;
private:
    std::shared_ptr<RocalCropParam> _crop_param;
    vx_array _src_roi_width,_src_roi_height;
    unsigned _layout, _roi_type;
};

