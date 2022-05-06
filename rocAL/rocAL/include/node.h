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
#include "graph.h"
#include "meta_data_graph.h"
#include "tensor.h"

class TensorNode
{
public:
    TensorNode(const std::vector<rocALTensor *> &inputs,const std::vector<rocALTensor *> &outputs) :
        _inputs(inputs),
        _outputs(outputs),
        _batch_size(outputs[0]->info().batch_size()) {}
    virtual ~TensorNode();
    void create(std::shared_ptr<Graph> graph);
    void update_parameters();
    std::vector<rocALTensor *> input() { return _inputs; };
    std::vector<rocALTensor *> output() { return _outputs; };
    void add_next(const std::shared_ptr<rocALTensor>& node) {} // To be implemented
    void add_previous(const std::shared_ptr<rocALTensor>& node) {} //To be implemented
    std::shared_ptr<Graph> graph() { return _graph; }
    bool _is_ssd = false;
    void set_meta_data(MetaDataBatch* meta_data_info){_meta_data_info = meta_data_info;}

    protected:
    virtual void create_node() = 0;
    virtual void update_node() = 0;
    virtual void update_src_roi();

    const std::vector<rocALTensor *> _inputs;
    const std::vector<rocALTensor *> _outputs;
    std::shared_ptr<Graph> _graph = nullptr;
    vx_array _src_tensor_roi = nullptr;
    vx_node _node = nullptr;
    size_t _batch_size;
    MetaDataBatch* _meta_data_info;
};

class Node
{
public:
    Node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) :
        _inputs(inputs),
        _outputs(outputs),
        _batch_size(outputs[0]->info().batch_size()) {}
    virtual ~Node();
    void create(std::shared_ptr<Graph> graph);
    void update_parameters();
    std::vector<Tensor*> input() { return _inputs; };
    std::vector<Tensor*> output() { return _outputs; };
    void add_next(const std::shared_ptr<Node>& node) {} // To be implemented
    void add_previous(const std::shared_ptr<Node>& node) {} //To be implemented
    std::shared_ptr<Graph> graph() { return _graph; }
    bool _is_ssd = false;
    void set_meta_data(MetaDataBatch* meta_data_info){_meta_data_info = meta_data_info;}
protected:
    virtual void create_node() = 0;
    virtual void update_node() = 0;
    virtual void update_src_roi();
    std::vector<Tensor*> _inputs;
    std::vector<Tensor*> _outputs;
    std::shared_ptr<Graph> _graph = nullptr;
    vx_array _src_roi_width = nullptr;
    vx_array _src_roi_height = nullptr;
    vx_node _node = nullptr;
    size_t _batch_size;
    MetaDataBatch* _meta_data_info;
};
