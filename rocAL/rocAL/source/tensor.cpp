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

#include <cstdio>
#if !ENABLE_HIP
#include <CL/cl.h>
#endif
#include <stdexcept>
#include <vx_ext_amd.h>
#include <cstring>
#include "commons.h"
#include "tensor.h"

vx_enum vx_mem_type(RocalMemType mem)
{
    switch(mem)
    {
        case RocalMemType::OCL:
        {
            return VX_MEMORY_TYPE_OPENCL;
        }
        break;
        case RocalMemType::HOST:
        {
            return VX_MEMORY_TYPE_HOST;
        }
        break;
        case RocalMemType::HIP:
        {
            return VX_MEMORY_TYPE_HIP;
        }
        break;
        default:
            throw std::runtime_error("Memory type not valid");
    }
}

vx_size tensor_data_size(RocalTensorDataType data_type)
{
    switch(data_type)
    {
        case RocalTensorDataType::FP32:
        {
            return sizeof(vx_float32);
        }
        break;
        case RocalTensorDataType::FP16:
        {
            return sizeof(vx_int16);
        }
        break;
        case RocalTensorDataType::UINT8:
        {
            return sizeof(vx_uint8);
        }
        break;
        default:
            throw std::runtime_error("tensor data_type not valid");
    }
}

bool operator==(const TensorInfo &rhs, const TensorInfo &lhs)
{
    return (rhs.width() == lhs.width() &&
            rhs.batch_size() == lhs.batch_size() &&
            rhs.mem_type() == lhs.mem_type() &&
            rhs.data_type() == lhs.data_type() &&
            rhs.channels() == lhs.channels() &&
            rhs.format() == lhs.format());
}

//******************************************NEW ROCAL TENSOR******************************************************
void rocALTensorInfo::reallocate_tensor_roi_buffers()
{
    _roi = std::vector<RocalROI>(_batch_size);
    // _roi_height = std::make_shared<std::vector<uint32_t>>(_batch_size);
    // _roi_width = std::make_shared<std::vector<uint32_t>>(_batch_size);
    _roi.resize(_batch_size);
    if(layout() == RocalTensorlayout::NCHW)
    {
        for (unsigned i = 0; i < _batch_size; i++)
        {
            _roi[i].x1 = 0;
            _roi[i].y1 = 0;
            _roi[i].x2 = _dims->at(3);
            _roi[i].y2= _dims->at(2);
        }
    }
    else if(layout() == RocalTensorlayout::NHWC)
    {
        for (unsigned i = 0; i < _batch_size; i++)
        {
            _roi[i].x1 = 0;
            _roi[i].y1 = 0;
            _roi[i].x2 = _dims->at(2);
            _roi[i].y2= _dims->at(1);
            std::cerr<<"\n _roi :: "<<_dims->at(0)<<" "<<_dims->at(1)<<" "<<_dims->at(2)<<" "<<_dims->at(3);
            std::cerr<<"\n checking "<<_roi[i].x2<<" "<<_roi[i].y2<<"\n";
        }
    }
}

rocALTensorInfo::rocALTensorInfo() : _type(Type::UNKNOWN),
                                _num_of_dims(0),
                                _dims(nullptr),
                                _mem_type(RocalMemType::HOST),
                                _data_type(RocalTensorDataType::FP32){}

rocALTensorInfo::rocALTensorInfo(
    unsigned num_of_dims,
    std::shared_ptr<std::vector<unsigned>> dims,
    RocalMemType mem_type,
    RocalTensorDataType data_type) : _type(Type::UNKNOWN),
                                _num_of_dims(num_of_dims),
                                _dims(dims),
                                _batch_size(dims->at(0)),
                                _mem_type(mem_type),
                                _data_type(data_type)
{
    vx_size data_size = tensor_data_size(data_type);
    unsigned alignpixels = TENSOR_WIDTH_ALIGNMENT; // Check if needed
    _data_size = data_size;
    for(unsigned i = 0; i < _num_of_dims; i++)
    {
        _data_size *= dims->at(i);
    }
    if(_num_of_dims <= 3)
        _is_image = false;
    // initializing each Tensor dimension in the batch with the maximum Tensor size, they'll get updated later during the runtime
    // Update this only if the tensor is image
}

void rocALTensor::update_tensor_roi(const std::vector<uint32_t> &width, const std::vector<uint32_t> &height)
{
    if(_info.is_image())
    {
        if (width.size() != height.size())
            THROW("Batch size of Tensor height and width info does not match")

        if (width.size() != info().batch_size())
            THROW("The batch size of actual Tensor height and width different from Tensor batch size " + TOSTR(width.size()) + " != " + TOSTR(info().batch_size()))

        std::cerr<<"\n rocALTensor::update_tensor_roi :: "<<_info.get_roi()[0].x1<<" "<<_info.get_roi()[0].y1<<" "<<_info.get_roi()[0].x2<<" "<<_info.get_roi()[0].y2;
        for (unsigned i = 0; i < info().batch_size(); i++)
        {
            if (width[i] > _info.max_width())
            {
                ERR("Given ROI width is larger than buffer width for tensor[" + TOSTR(i) + "] " + TOSTR(width[i]) + " > " + TOSTR(_info.max_width()))
                _info.set_roi_width()->at(i).x2= _info.max_width();
            }
            else
            {
                _info.set_roi_width()->at(i).x2 = width[i];
            }

            if (height[i] > _info.max_height())
            {
                ERR("Given ROI height is larger than buffer with for tensor[" + TOSTR(i) + "] " + TOSTR(height[i]) + " > " + TOSTR(_info.max_height()))
                _info.set_roi_height()->at(i).y2 = _info.max_height();

            }
            else
            {
                _info.set_roi_height()->at(i).y2 =height[i];

            }
        }
        static int i=0;
        std::cerr<<"\n"<<" rocALTensor::update_tensor_roi :: "<<_info.get_roi()[0].x1<<" "<<_info.get_roi()[0].y1<<" "<<_info.get_roi()[0].x2 <<" "<<_info.get_roi()[0].y2<<"\n";
    }
}

rocALTensor::~rocALTensor()
{
    vxReleaseTensor(&_vx_handle);
}

//! Converts the Rocal data_type to OpenVX
vx_enum interpret_tensor_data_type(RocalTensorDataType data_type)
{
    switch (data_type)
    {
        case RocalTensorDataType::FP32:
            return VX_TYPE_FLOAT32;
        case RocalTensorDataType::FP16:
            return VX_TYPE_FLOAT16;
        case RocalTensorDataType::UINT8:
            return VX_TYPE_UINT8;
        default:
            THROW("Unsupported Tensor type " + TOSTR(data_type))
    }
}

rocALTensor::rocALTensor(const rocALTensorInfo &tensor_info) : _info(tensor_info)
{
    _info._type = rocALTensorInfo::Type::UNKNOWN;
    _mem_handle = nullptr;
}

int rocALTensor::create_virtual(vx_context context, vx_graph graph)
{
    if (_vx_handle)
        return -1;

    _context = context;

    // create a virtual Tensor as the output Tensor for this node
    // vx_size dims[4];
    // dims[0] = (vx_size)_info.batch_size();
    // dims[1] = (vx_size)_info.height();
    // dims[2] = (vx_size)_info.width();
    // dims[3] = (vx_size)_info.channels();
    // shobi find a better way to convert from unsigned to size_t
    vx_size dims[4];
    for(unsigned i = 0; i < _info.num_of_dims(); i++)
    {
        dims[i] = _info.dims()->at(i);
    }
    _vx_handle = vxCreateVirtualTensor(graph, _info.num_of_dims(), dims, interpret_tensor_data_type(_info.data_type()), 0);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateVirtualTensor(input:[" + TOSTR(_info.max_width()) + "W" + TOSTR(_info.max_height()) + "H" + "]): failed " + TOSTR(status))

    _info._type = rocALTensorInfo::Type::VIRTUAL;
    return 0;
}

int rocALTensor::create_from_handle(vx_context context)
{
    if (_vx_handle)
    {
        WRN("Tensor object create method is already called ")
        return -1;
    }
    // if (_info.height() == 0 || _info.width() == 0 || _info.channels() == 0 || _info.batch_size() == 0)
    //     THROW("Invalid tensor dimension" + TOSTR(_info.width()) + " x " + TOSTR(_info.height()) + " x " + TOSTR(_info.channels()) + " x " + TOSTR(_info.batch_size()));

    _context = context;
    // bool nhwc = true;
    vx_enum tens_data_type = interpret_tensor_data_type(_info.data_type());
    // vx_size dims[4];
    // if (nhwc)
    // {
    //     dims[0] = (vx_size)_info.batch_size();
    //     dims[1] = (vx_size)_info.height();
    //     dims[2] = (vx_size)_info.width();
    //     dims[3] = (vx_size)_info.channels();
    // }
    // else
    // {
    //     dims[0] = (vx_size)_info.width();
    //     dims[1] = (vx_size)_info.height();
    //     dims[2] = (vx_size)_info.channels();
    //     dims[3] = (vx_size)_info.batch_size();
    // }

    vx_size stride[_info.num_of_dims()];
    void *ptr[1] = {nullptr};
    bool nhwc = ((_info.layout() == RocalTensorlayout::NHWC) ? true: false);

    stride[0] = tensor_data_size(_info.data_type());
    for(unsigned i = 1; i < _info.num_of_dims(); i++)
    {
        stride[i] = stride[i - 1] * _info.dims()->at(i - 1);
    }
    vx_status status;
    // shobi find a better way to convert from unsigned to size_t
    vx_size dims[4];
    for(unsigned i = 0; i < _info.num_of_dims(); i++)
    {
        dims[i] = _info.dims()->at(i);
    }
    // dims[0] = _info.dims()->at(0);
    // dims[1] = _info.dims()->at(1);
    // dims[2] = _info.dims()->at(2);
    // dims[3] = _info.dims()->at(3);
    // std::cerr<<"\n dims in local "<<dims[0]<<" "<<dims[1]<<" "<<dims[2]<<" "<<dims[3];
    _vx_handle = vxCreateTensorFromHandle(_context, _info.num_of_dims(), dims, tens_data_type, 0, stride, ptr, vx_mem_type(_info._mem_type));
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(input: failed " + TOSTR(status))
    _info._type = rocALTensorInfo::Type::HANDLE;
    // shobi to be checked
    // if(_info._data_size == 0)
    //     _info._data_size = stride[3] * _info.channels(); // since data size is set while initializing info
    return 0;
}

int rocALTensor::create(vx_context context)
{
    if (_vx_handle)
        return -1;

    _context = context;

    vx_status status;
    // vx_size dims[4]; // = {(vx_size)_info.width(), (vx_size)_info.height(), (vx_size)_info.channels(), (vx_size)_info.batch_size()};
    // dims[0] = (vx_size)_info.batch_size();
    // dims[1] = (vx_size)_info.max_height();
    // dims[2] = (vx_size)_info.max_width();
    // dims[3] = (vx_size)_info.channels();
    vx_enum tens_data_type = interpret_tensor_data_type(_info.data_type());
    _vx_handle = vxCreateTensor(context, _info.num_of_dims(),(vx_size*) _info.dims()->data(), tens_data_type, 0);
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateTensor(input: failed " + TOSTR(status))
    _info._type = rocALTensorInfo::Type::REGULAR;
    return 0;
}
#if !ENABLE_HIP
unsigned rocALTensor::copy_data(cl_command_queue queue, unsigned char *user_buffer, bool sync)
{
    if (_info._type != rocALTensorInfo::Type::HANDLE)
        return 0;

    // unsigned size = _info.stride() * _info.height() * _info.channels() * _info.batch_size();

    if (_info._mem_type == RocalMemType::OCL)
    {

        cl_int status;
        if ((status = clEnqueueReadBuffer(queue,
                                          (cl_mem)_mem_handle,
                                          sync ? (CL_TRUE) : CL_FALSE,
                                          0,
                                          _info.data_size(),
                                          user_buffer,
                                          0, nullptr, nullptr)) != CL_SUCCESS)
            THROW("clEnqueueReadBuffer failed: " + TOSTR(status))
    }
    else
    {
        memcpy(user_buffer, _mem_handle, _info.data_size());
    }
    return _info.data_size();
}
unsigned rocALTensor::copy_data(cl_command_queue queue, cl_mem user_buffer, bool sync)
{
    return 0;
}

#else
unsigned rocALTensor::copy_data(hipStream_t stream, unsigned char* user_buffer, bool sync)
{
    if(_info._type != rocALTensorInfo::Type::HANDLE)
        return 0;

    // unsigned size = _info.width() *
    //                 _info.height_batch() *
    //                 _info.color_plane_count();

    if (_info._mem_type == RocalMemType::HIP)
    {
        // copy from device to host
        hipError_t status;
        if ((status = hipMemcpyDtoHAsync((void *)user_buffer, _mem_handle, _info.data_size(), stream)))
            THROW("copy_data::hipMemcpyDtoHAsync failed: " + TOSTR(status))
        if (sync) {
            if ((status =hipStreamSynchronize(stream)))
                THROW("copy_data::hipStreamSynchronize failed: " + TOSTR(status))
        }

    }
    else
    {
        memcpy(user_buffer, _mem_handle, size);
    }
    return size;
}
unsigned rocALTensor::copy_data(hipStream_t stream, void* hip_memory, bool sync)
{
    // todo:: copy from host to device
    return 0;
}
#endif

int rocALTensor::swap_handle(void *handle)
{
    vx_status status;
    if ((status = vxSwapTensorHandle(_vx_handle, handle, nullptr)) != VX_SUCCESS)
    {
        ERR("Swap handles failed for tensor" + TOSTR(status));
        return -1;
    }

    // Updating the buffer pointer as well,
    // user might want to copy directly using it
    _mem_handle = handle;
    return 0;
}

//******************************************OLD ROCAL TENSOR******************************************************

uint32_t *TensorInfo::get_roi_width() const
{
    return _roi_width->data();
}

uint32_t *TensorInfo::get_roi_height() const
{
    return _roi_height->data();
}

const std::vector<uint32_t> &TensorInfo::get_roi_width_vec() const
{
    return *_roi_width;
}

const std::vector<uint32_t> &TensorInfo::get_roi_height_vec() const
{
    return *_roi_height;
}

unsigned TensorInfo::get_roi_width(int batch_idx) const
{
    if ((unsigned)batch_idx >= _roi_width->size())
        THROW("Accessing roi width out of tensor range")
    if (!_roi_width->at(batch_idx))
        THROW("Accessing uninitialized int parameter associated with image width")
    return _roi_width->at(batch_idx);
}

unsigned TensorInfo::get_roi_height(int batch_idx) const
{
    if ((unsigned)batch_idx >= _roi_height->size())
        THROW("AAccessing roi width out of tensor range")
    if (!_roi_height->at(batch_idx))
        THROW("Accessing uninitialized int parameter associated with image height")
    return _roi_height->at(batch_idx);
}
void TensorInfo::reallocate_tensor_roi_buffers()
{
    _roi_height = std::make_shared<std::vector<uint32_t>>(_batch_size);
    _roi_width = std::make_shared<std::vector<uint32_t>>(_batch_size);
    for (unsigned i = 0; i < _batch_size; i++)
    {
        _roi_height->at(i) = height();
        _roi_width->at(i) = width();
    }
}
TensorInfo::TensorInfo() : _type(Type::UNKNOWN),
                           _width(0),
                           _height(0),
                           _batch_size(1),
                           _channels(1),
                           _data_size(0),
                           _mem_type(RocalMemType::HOST),
                           _color_fmt(RocalColorFormat::U8),
                           _data_type(RocalTensorDataType::FP32),
                           _format(RocalTensorlayout::NHWC){}

TensorInfo::TensorInfo(
    unsigned width_,
    unsigned height_,
    unsigned batches_,
    unsigned channels_,
    RocalMemType mem_type_,
    RocalColorFormat col_fmt_,
    RocalTensorDataType data_type,
    RocalTensorlayout tensor_format) : _type(Type::UNKNOWN),
                                      _width(width_),
                                      _height(height_),
                                      _batch_size(batches_),
                                      _channels(channels_),
                                      _data_size(width_ * height_ * _batch_size * channels_),
                                      _mem_type(mem_type_),
                                      _color_fmt(col_fmt_),
                                      _data_type(data_type),
                                      _format(tensor_format)
{
    vx_size pix_size = tensor_data_size(data_type);
    unsigned alignpixels = TENSOR_WIDTH_ALIGNMENT;
    std::cerr<<"################ TensorInfo :: "<<_width <<" "<<_height<<"\n";
    if (alignpixels == 0)
        _stride = width_ * pix_size;
    else
        _stride = ((width_ + alignpixels - 1) & ~(alignpixels - 1)) * pix_size;
    _data_size = _stride * height_ * _batch_size * _channels;
    // initializing each Tensor dimension in the batch with the maximum Tensor size, they'll get updated later during the runtime
    reallocate_tensor_roi_buffers();
}

void Tensor::update_tensor_roi(const std::vector<uint32_t> &width, const std::vector<uint32_t> &height)
{
    if (width.size() != height.size())
        THROW("Batch size of Tensor height and width info does not match")

    if (width.size() != info().batch_size())
        THROW("The batch size of actual Tensor height and width different from Tensor batch size " + TOSTR(width.size()) + " != " + TOSTR(info().batch_size()))
    if (!_info._roi_width || !_info._roi_height)
        THROW("ROI width or ROI height vector not created")
    for (unsigned i = 0; i < info().batch_size(); i++)
    {

        if (width[i] > _info.width())
        {
            ERR("Given ROI width is larger than buffer width for tensor[" + TOSTR(i) + "] " + TOSTR(width[i]) + " > " + TOSTR(_info.width()))
            _info._roi_width->at(i) = _info.width();
        }
        else
        {
            _info._roi_width->at(i) = width[i];
        }

        if (height[i] > _info.height())
        {
            ERR("Given ROI height is larger than buffer with for tensor[" + TOSTR(i) + "] " + TOSTR(height[i]) + " > " + TOSTR(_info.height()))
            _info._roi_height->at(i) = _info.height();
        }
        else
        {
            _info._roi_height->at(i) = height[i];
        }
    }
}

Tensor::~Tensor()
{
    vxReleaseTensor(&_vx_handle);
}

//! Converts the Rocal data_type to OpenVX
// vx_enum interpret_tensor_data_type(RocalTensorDataType data_type)
// {
//     switch (data_type)
//     {
//         case RocalTensorDataType::FP32:
//             return VX_TYPE_FLOAT32;
//         case RocalTensorDataType::FP16:
//             return VX_TYPE_FLOAT16;
//         case RocalTensorDataType::UINT8:
//             return VX_TYPE_UINT8;
//         default:
//             THROW("Unsupported Tensor type " + TOSTR(data_type))
//     }
// }

Tensor::Tensor(const TensorInfo &tensor_info) : _info(tensor_info)
{
    _info._type = TensorInfo::Type::UNKNOWN;
    _mem_handle = nullptr;
}

int Tensor::create_virtual(vx_context context, vx_graph graph)
{
    if (_vx_handle)
        return -1;

    _context = context;

    // create a virtual Tensor as the output Tensor for this node
    vx_size dims[4];
    dims[0] = (vx_size)_info.batch_size();
    dims[1] = (vx_size)_info.height();
    dims[2] = (vx_size)_info.width();
    dims[3] = (vx_size)_info.channels();
    _vx_handle = vxCreateVirtualTensor(graph, 4, dims, interpret_tensor_data_type(_info.data_type()), 0);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateVirtualTensor(input:[" + TOSTR(_info.width()) + "W" + TOSTR(_info.height()) + "H" + "]): failed " + TOSTR(status))

    _info._type = TensorInfo::Type::VIRTUAL;
    return 0;
}

int Tensor::create_from_handle(vx_context context)
{
    if (_vx_handle)
    {
        WRN("Tensor object create method is already called ")
        return -1;
    }
    if (_info.height() == 0 || _info.width() == 0 || _info.channels() == 0 || _info.batch_size() == 0)
        THROW("Invalid tensor dimension" + TOSTR(_info.width()) + " x " + TOSTR(_info.height()) + " x " + TOSTR(_info.channels()) + " x " + TOSTR(_info.batch_size()));

    _context = context;
    bool nhwc = true;
    vx_enum tens_data_type = interpret_tensor_data_type(_info.data_type());
    vx_size dims[4];
    if (nhwc)
    {
        dims[0] = (vx_size)_info.batch_size();
        dims[1] = (vx_size)_info.height();
        dims[2] = (vx_size)_info.width();
        dims[3] = (vx_size)_info.channels();
    }
    else
    {
        dims[0] = (vx_size)_info.width();
        dims[1] = (vx_size)_info.height();
        dims[2] = (vx_size)_info.channels();
        dims[3] = (vx_size)_info.batch_size();
    }

    vx_size stride[4];
    void *ptr[1] = {nullptr};

    stride[0] = tensor_data_size(_info.data_type());

    vx_uint32 alignpixels = TENSOR_WIDTH_ALIGNMENT;
    if (nhwc)
    {
        if (alignpixels == 0)
            stride[1] = _info.batch_size() * stride[0];
        else
            stride[1] = ((_info.batch_size() + alignpixels - 1) & ~(alignpixels - 1)) * stride[0];
        stride[2] = _info.height() * stride[1];
        stride[3] = _info.width() * stride[2];
    }
    else
    {
        if (alignpixels == 0)
            stride[1] = _info.width() * stride[0];
        else
            stride[1] = ((_info.width() + alignpixels - 1) & ~(alignpixels - 1)) * stride[0];
        stride[2] = _info.height() * stride[1];
        stride[3] = _info.channels() * stride[2];
    }

    vx_status status;
    _vx_handle = vxCreateTensorFromHandle(_context, 4, dims, tens_data_type, 0, stride, ptr, vx_mem_type(_info._mem_type));
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(input:[" + TOSTR(_info.width()) + "x" + TOSTR(_info.height()) + "]): failed " + TOSTR(status))
    _info._type = TensorInfo::Type::HANDLE;
    if(_info._data_size == 0)
        _info._data_size = stride[3] * _info.channels(); // since data size is set while initializing info
    return 0;
}
int Tensor::create(vx_context context)
{
    if (_vx_handle)
        return -1;

    _context = context;

    vx_status status;
    vx_size dims[4]; // = {(vx_size)_info.width(), (vx_size)_info.height(), (vx_size)_info.channels(), (vx_size)_info.batch_size()};
    dims[0] = (vx_size)_info.batch_size();
    dims[1] = (vx_size)_info.height();
    dims[2] = (vx_size)_info.width();
    dims[3] = (vx_size)_info.channels();
    vx_enum tens_data_type = interpret_tensor_data_type(_info.data_type());
    _vx_handle = vxCreateTensor(context, 4, dims, tens_data_type, 0);
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateTensor(input:[" + TOSTR(_info.width()) + "x" + TOSTR(_info.height()) + "x" + TOSTR(_info.channels()) + "]): failed " + TOSTR(status))
    _info._type = TensorInfo::Type::REGULAR;
    return 0;
}
#if !ENABLE_HIP
unsigned Tensor::copy_data(cl_command_queue queue, unsigned char *user_buffer, bool sync)
{
    if (_info._type != TensorInfo::Type::HANDLE)
        return 0;

    unsigned size = _info.stride() * _info.height() * _info.channels() * _info.batch_size();

    if (_info._mem_type == RocalMemType::OCL)
    {

        cl_int status;
        if ((status = clEnqueueReadBuffer(queue,
                                          (cl_mem)_mem_handle,
                                          sync ? (CL_TRUE) : CL_FALSE,
                                          0,
                                          size,
                                          user_buffer,
                                          0, nullptr, nullptr)) != CL_SUCCESS)
            THROW("clEnqueueReadBuffer failed: " + TOSTR(status))
    }
    else
    {
        memcpy(user_buffer, _mem_handle, size);
    }
    return size;
}
unsigned Tensor::copy_data(cl_command_queue queue, cl_mem user_buffer, bool sync)
{
    return 0;
}

#else
unsigned Tensor::copy_data(hipStream_t stream, unsigned char* user_buffer, bool sync)
{
    if(_info._type != TensorInfo::Type::HANDLE)
        return 0;

    unsigned size = _info.width() *
                    _info.height_batch() *
                    _info.color_plane_count();

    if (_info._mem_type == RocalMemType::HIP)
    {
        // copy from device to host
        hipError_t status;
        if ((status = hipMemcpyDtoHAsync((void *)user_buffer, _mem_handle, size, stream)))
            THROW("copy_data::hipMemcpyDtoHAsync failed: " + TOSTR(status))
        if (sync) {
            if ((status =hipStreamSynchronize(stream)))
                THROW("copy_data::hipStreamSynchronize failed: " + TOSTR(status))
        }

    }
    else
    {
        memcpy(user_buffer, _mem_handle, size);
    }
    return size;
}
unsigned Tensor::copy_data(hipStream_t stream, void* hip_memory, bool sync)
{
    // todo:: copy from host to device
    return 0;
}
#endif

int Tensor::swap_handle(void *handle)
{
    vx_status status;
    if ((status = vxSwapTensorHandle(_vx_handle, handle, nullptr)) != VX_SUCCESS)
    {
        ERR("Swap handles failed for tensor" + TOSTR(status));
        return -1;
    }

    // Updating the buffer pointer as well,
    // user might want to copy directly using it
    _mem_handle = handle;
    return 0;
}