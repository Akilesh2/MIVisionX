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

#include <VX/vx.h>
#include <VX/vx_types.h>
#include <vector>
#include <cstring>
#include <array>
#include <queue>
#include <memory>
#if ENABLE_HIP
#include "hip/hip_runtime.h"
#include "device_manager_hip.h"
#else
#include "device_manager.h"
#endif
#include "commons.h"
// to align tensor width
#define TENSOR_WIDTH_ALIGNMENT   0 // todo:: check to see if we need this

/*! \brief Converts Rocal Memory type to OpenVX memory type
 *
 * @param mem input Rocal type
 * @return the OpenVX type associated with input argument
 */
extern vx_enum vx_mem_type(RocalMemType mem);

struct rocALTensorInfo
{
    friend struct rocALTensor;
    enum class Type
    {
        UNKNOWN = -1,
        REGULAR = 0,
        VIRTUAL = 1,
        HANDLE  = 2
    };
    // Default constructor
    rocALTensorInfo();

    //! Initializer constructor with only fields common to all types (Image/ Video / Audio)
    rocALTensorInfo(unsigned num_of_dims,
                    std::shared_ptr<std::vector<unsigned>> dims,
                    RocalMemType mem_type,
                    RocalTensorDataType data_type);

    // Setting properties required for Image / Video
    void set_roi_type(RocalROIType roi_type) { _roi_type = roi_type; }
    void set_data_type(RocalTensorDataType data_type) { _data_type = _data_type; }

    void set_tensor_layout(RocalTensorlayout layout)
    {
        if(layout != RocalTensorlayout::NONE)
        {
            _is_image = true;
            if(layout == RocalTensorlayout::NHWC)
            {
                _max_width = _dims->at(2);
                _max_height = _dims->at(1);
            }
            else if(layout == RocalTensorlayout::NCHW)
            {
                _max_width = _dims->at(3);
                _max_height = _dims->at(2);
            }
            reallocate_tensor_roi_buffers();
        }
        else
        {
            _max_width = _dims->at(1);
            _max_height = _dims->at(2);
            std::cerr<<"\n Setting _max_width :: "<<_max_width<<"\t _max_height :: "<<_max_height;
        }
        _layout = layout;
    }
    void set_color_format(RocalColorFormat color_format) { _color_format = color_format; }
    // void set _data_type(Rocal)

    unsigned num_of_dims() const { return _num_of_dims; }
    unsigned batch_size() const { return _batch_size; }
    unsigned data_size() const { return _data_size; }
    unsigned max_width() const { return _max_width; }
    unsigned max_height() const { return _max_height; }
    std::shared_ptr<std::vector<unsigned>> dims() const { return _dims; }
    RocalMemType mem_type() const { return _mem_type; }
    RocalROIType roi_type() const { return _roi_type; }
    RocalTensorDataType data_type() const { return _data_type; }
    RocalTensorlayout layout() const { return _layout; }
    std::vector<RocalROI> get_roi() const { return _roi; }
    RocalColorFormat color_format() const {return _color_format; }
    Type type() const { return _type; }
    bool is_image() const { return _is_image; }
    unsigned data_type_size()
    {
        if(_data_type == RocalTensorDataType::FP32)
        {
            _data_type_size = sizeof(vx_float32);
        }
        else if(_data_type == RocalTensorDataType::FP16)
        {
            _data_type_size = sizeof(vx_int16); // have to change this to float 16
        }
        else if(_data_type == RocalTensorDataType::UINT8)
        {
            _data_type_size = sizeof(vx_uint8);
        }
        return _data_type_size;
    }


private:
    Type _type = Type::UNKNOWN;//!< tensor type, whether is virtual tensor, created from handle or is a regular tensor
    unsigned _num_of_dims;
    std::shared_ptr<std::vector<unsigned>> _dims;
    RocalMemType _mem_type;
    RocalROIType _roi_type;
    RocalTensorDataType _data_type = RocalTensorDataType::FP32;
    RocalTensorlayout _layout = RocalTensorlayout::NCHW;
    RocalColorFormat _color_format;
    std::vector<RocalROI> _roi;
    unsigned _data_type_size;
    unsigned _batch_size;
    unsigned _data_size = 0;
    unsigned _max_width, _max_height;
    bool _is_image = false;
    void reallocate_tensor_roi_buffers();
};

bool operator==(const rocALTensorInfo& rhs, const rocALTensorInfo& lhs);
struct rocALTensor
{
    int swap_handle(void* handle);

    const rocALTensorInfo& info() { return _info; }
    //! Default constructor
    rocALTensor() = delete;
    void* buffer() { return _mem_handle; }
    vx_tensor handle() { return _vx_handle; }
    vx_context context() { return _context; }
#if !ENABLE_HIP
    unsigned copy_data(cl_command_queue queue, unsigned char* user_buffer, bool sync);
    unsigned copy_data(cl_command_queue queue, cl_mem user_buffer, bool sync);
#else
    unsigned copy_data(hipStream_t stream, unsigned char* user_buffer, bool sync);
    unsigned copy_data(hipStream_t stream, void* hip_memory, bool sync);
#endif
    //! Default destructor
    /*! Releases the OpenVX Tensor object */
    ~rocALTensor();

    //! Constructor accepting the image information as input
    explicit rocALTensor(const rocALTensorInfo& tensor_info);

    int create(vx_context context);
    void update_tensor_roi(const std::vector<uint32_t> &width, const std::vector<uint32_t> &height);
    void reset_tensor_roi() { _info.reallocate_tensor_roi_buffers(); }
    // create_from_handle() no internal memory allocation is done here since tensor's handle should be swapped with external buffers before usage
    int create_from_handle(vx_context context);
    int create_virtual(vx_context context, vx_graph graph);

private:
    vx_tensor _vx_handle = nullptr;//!< The OpenVX image
    void* _mem_handle = nullptr;//!< Pointer to the image's internal buffer (opencl or host)
    rocALTensorInfo _info;//!< The structure holding the info related to the stored OpenVX image
    vx_context _context = nullptr;
};


/*! \brief Holds the information about an OpenVX Tensor */

struct TensorInfo
{
    friend struct Tensor;
    enum class Type
    {
        UNKNOWN = -1,
        REGULAR =0,
        VIRTUAL = 1,
        HANDLE =2
    };
    //! Default constructor,
    /*! initializes memory type to host  */
    TensorInfo();

    //! Initializer constructor
    TensorInfo(
        unsigned width_,
        unsigned height_,
        unsigned batches_,
        unsigned channels_,
        RocalMemType mem_type_,
        RocalColorFormat color_format,
        RocalTensorDataType data_type,
        RocalTensorlayout   tensor_format);

    unsigned width() const { return _width; }
    unsigned height_batch() const {return _height * _batch_size; }
    unsigned height() const { return _height; }
    unsigned channels() const { return _channels; }
    unsigned stride() const { return _stride; }
    unsigned height_single() const { return _height;}
    void width(unsigned width) { _width = width; }
    void height(unsigned height) { _height = height; }
    void batch_size(unsigned batch_size) { _batch_size = batch_size; }
    void channels(unsigned channels) { _channels = channels; }
    void format(RocalTensorlayout format) { _format = format; }
    void color_format(RocalColorFormat color_fmt)  { _color_fmt = color_fmt; }
    void data_type(RocalTensorDataType data_type)  { _data_type = data_type; }
    Type type() const { return _type; }
    RocalTensorDataType data_type() const { return _data_type;}
    unsigned batch_size() const {return _batch_size;}
    RocalMemType mem_type() const { return _mem_type; }
    RocalTensorlayout format() const { return _format; }
    unsigned data_size() const { return _data_size; }
    RocalColorFormat color_format() const {return _color_fmt; }
    unsigned color_plane_count() const { return _color_planes; }
    unsigned get_roi_width(int batch_idx) const;
    unsigned get_roi_height(int batch_idx) const;
    uint32_t * get_roi_width() const;
    uint32_t * get_roi_height() const;
    const std::vector<uint32_t>& get_roi_width_vec() const;
    const std::vector<uint32_t>& get_roi_height_vec() const;
    bool is_image() const { return _is_image; }
    size_t data_type_size()
    {
        if(_data_type == RocalTensorDataType::FP32)
        {
            _data_type_size = sizeof(vx_float32);
        }
        else if(_data_type == RocalTensorDataType::FP16)
        {
            _data_type_size = sizeof(vx_int16); // have to change this to float 16
        }
        else if(_data_type == RocalTensorDataType::UINT8)
        {
            _data_type_size = sizeof(vx_uint8);
        }
        return _data_type_size;
    }
private:
    Type _type = Type::UNKNOWN;//!< image type, whether is virtual image, created from handle or is a regular image
    unsigned _width;//!< image width for a single image in the batch
    unsigned _height;//!< image height for a single image in the batch
    unsigned _batch_size;//!< the batch size (images in the batch are stacked on top of each other)
    unsigned _channels;//!< number of channels
    unsigned _data_size;//!< total size of the memory needed to keep the image's data in bytes including all planes
    RocalMemType _mem_type;//!< memory type, currently either OpenCL or Host
    RocalColorFormat _color_fmt;//!< color format of the image
    RocalTensorDataType _data_type = RocalTensorDataType::FP32;
    RocalTensorlayout _format = RocalTensorlayout::NCHW;
    unsigned _color_planes;//!< number of color planes
    unsigned _stride;//!< if different from width
    std::shared_ptr<std::vector<uint32_t>> _roi_width;//!< The actual image width stored in the buffer, it's always smaller than _width/_batch_size. It's created as a vector of pointers to integers, so that if it's passed from one image to another and get updated by one and observed for all.
    std::shared_ptr<std::vector<uint32_t>> _roi_height;//!< The actual image height stored in the buffer, it's always smaller than _height. It's created as a vector of pointers to integers, so that if it's passed from one image to another and get updated by one changes can be observed for all.

    void reallocate_tensor_roi_buffers();
    bool _is_image = false;
    size_t _data_type_size;
};

bool operator==(const TensorInfo& rhs, const TensorInfo& lhs);

/*! \brief Holds an OpenVX tensor and it's info
*
* Keeps the information about the ROCAL tensor that can be queried using OVX API as well,
* but for simplicity and ease of use, they are kept in separate fields
*/
struct Tensor
{
    int swap_handle(void* handle);

    const TensorInfo& info() { return _info; }
    //! Default constructor
    Tensor() = delete;
    void* buffer() { return _mem_handle; }
    vx_tensor handle() { return _vx_handle; }
    vx_context context() { return _context; }
#if !ENABLE_HIP
    unsigned copy_data(cl_command_queue queue, unsigned char* user_buffer, bool sync);
    unsigned copy_data(cl_command_queue queue, cl_mem user_buffer, bool sync);
#else
    unsigned copy_data(hipStream_t stream, unsigned char* user_buffer, bool sync);
    unsigned copy_data(hipStream_t stream, void* hip_memory, bool sync);
#endif
    //! Default destructor
    /*! Releases the OpenVX Tensor object */
    ~Tensor();

    //! Constructor accepting the image information as input
    explicit Tensor(const TensorInfo& tensor_info);

    int create(vx_context context);
    void update_tensor_roi(const std::vector<uint32_t> &width, const std::vector<uint32_t> &height);
    void reset_tensor_roi() { _info.reallocate_tensor_roi_buffers(); }
    // create_from_handle() no internal memory allocation is done here since tensor's handle should be swapped with external buffers before usage
    int create_from_handle(vx_context context);
    int create_virtual(vx_context context, vx_graph graph);

private:
    vx_tensor _vx_handle = nullptr;//!< The OpenVX image
    void* _mem_handle = nullptr;//!< Pointer to the image's internal buffer (opencl or host)
    TensorInfo _info;//!< The structure holding the info related to the stored OpenVX image
    vx_context _context = nullptr;
};



