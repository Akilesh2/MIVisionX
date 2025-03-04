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

#pragma once
#include <set>
#include <memory>
#include "bounding_box_graph.h"
#include "meta_data.h"
#include "node.h"
#include "node_resize.h"
#include "parameter_vx.h"
class ResizeMetaNode:public MetaNode
{
    public:
        ResizeMetaNode() {};
        void update_parameters(MetaDataBatch* input_meta_data)override;
        std::shared_ptr<ResizeNode> _node = nullptr;
    private:
        void initialize();
        vx_array _src_width, _src_height;
        std::vector<uint> _src_width_val, _src_height_val;
        unsigned int _dst_width, _dst_height;
        float _dst_to_src_width_ratio, _dst_to_src_height_ratio;
};
