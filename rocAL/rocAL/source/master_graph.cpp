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
#if !ENABLE_HIP
#include <CL/cl.h>
#endif
#include <vx_ext_amd.h>
#include <VX/vx_types.h>
#include <cstring>
#include <sched.h>
#include <half.hpp>
#include "master_graph.h"
#include "parameter_factory.h"
#include "ocl_setup.h"
#include "meta_data_reader_factory.h"
#include "meta_data_graph_factory.h"
// #include "randombboxcrop_meta_data_reader_factory.h"

using half_float::half;

#if ENABLE_SIMD
#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif
#endif

#if ENABLE_HIP
#include <rocal_hip_kernels.h>
#endif

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char* string)
{
    size_t len = strnlen(string, MAX_STRING_LENGTH);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}


auto get_ago_affinity_info = []
    (RocalAffinity rocal_affinity,
     int cpu_id,
     int gpu_id)
{
    AgoTargetAffinityInfo affinity;
    switch(rocal_affinity) {
        case RocalAffinity::GPU:
            affinity.device_type =  AGO_TARGET_AFFINITY_GPU;
            affinity.device_info = (gpu_id >=0 && gpu_id <=9)? gpu_id : 0;
            break;
        case RocalAffinity::CPU:
            affinity.device_type = AGO_TARGET_AFFINITY_CPU;
            affinity.device_info = (cpu_id >=0 && cpu_id <=9)? cpu_id : 0;
            break;
        default:
            throw std::invalid_argument("Unsupported affinity");
    }
    return affinity;
};

MasterGraph::~MasterGraph()
{
    release();
}

MasterGraph::MasterGraph(size_t batch_size, RocalAffinity affinity, int gpu_id, size_t cpu_threads,  size_t prefetch_queue_depth, RocalTensorDataType output_tensor_data_type):
        _ring_buffer(OUTPUT_RING_BUFFER_DEPTH),
        _output_tensor(nullptr),
        _graph(nullptr),
        _affinity(affinity),
        _gpu_id(gpu_id),
        _convert_time("Conversion Time", DBG_TIMING),
        _user_batch_size(batch_size),
        _cpu_threads(cpu_threads),
#if ENABLE_HIP
        _mem_type ((_affinity == RocalAffinity::GPU) ? RocalMemType::HIP : RocalMemType::HOST),
#else
        _mem_type ((_affinity == RocalAffinity::GPU) ? RocalMemType::OCL : RocalMemType::HOST),
#endif
        _process_time("Process Time", DBG_TIMING),
        _first_run(true),
        _processing(false),
        _internal_batch_size(compute_optimum_internal_batch_size(batch_size, affinity)),
        _user_to_internal_batch_ratio (_user_batch_size/_internal_batch_size),
        _prefetch_queue_depth(prefetch_queue_depth),
        _out_data_type(output_tensor_data_type)
{
    try {
        vx_status status;
        vxRegisterLogCallback(NULL, log_callback, vx_false_e);
        _context = vxCreateContext();
        vxRegisterLogCallback(_context, log_callback, vx_false_e);
        auto vx_affinity = get_ago_affinity_info(_affinity, 0, gpu_id);
        if ((status = vxGetStatus((vx_reference) _context)) != VX_SUCCESS)
            THROW("vxCreateContext failed" + TOSTR(status))

        if(affinity == RocalAffinity::GPU)
        {
#if !ENABLE_HIP
            if (_mem_type == RocalMemType::OCL){
                cl_context _cl_context = nullptr;
                cl_device_id _cl_device_id = nullptr;
                get_device_and_context(gpu_id, &_cl_context, &_cl_device_id, CL_DEVICE_TYPE_GPU);
                if((status = vxSetContextAttribute(_context,
                        VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT,
                        &_cl_context, sizeof(cl_context)) != VX_SUCCESS))
                    THROW("vxSetContextAttribute for CL_CONTEXT failed " + TOSTR(status))
            }
#else
            if (_mem_type == RocalMemType::HIP) {
                hipError_t err = hipInit(0);
                if (err != hipSuccess) {
                    THROW("ERROR: hipInit(0) => %d (failed)" + TOSTR(err));
                }
                int hip_num_devices = -1;
                err = hipGetDeviceCount(&hip_num_devices);
                if (err != hipSuccess) {
                    THROW("ERROR: hipGetDeviceCount() => %d (failed)" + TOSTR(err));
                }
                if (gpu_id < hip_num_devices) {
                    //set the device forcontext if specified.
                    int hipDevice = gpu_id;
                    if((status = vxSetContextAttribute(_context,
                            VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE,
                            &hipDevice, sizeof(hipDevice)) != VX_SUCCESS))
                        THROW("vxSetContextAttribute for hipDevice(%d) failed " + TOSTR(hipDevice) + TOSTR(status))
                }
            }
#endif
        }

        // Setting attribute to run on CPU or GPU should be called before load kernel module
        if ((status = vxSetContextAttribute(_context,
                                            VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY,
                                            &vx_affinity,
                                            sizeof(vx_affinity))) != VX_SUCCESS)
            THROW("vxSetContextAttribute for AMD_AFFINITY failed " + TOSTR(status))

        // loading OpenVX RPP modules
        if ((status = vxLoadKernels(_context, "vx_rpp")) != VX_SUCCESS)
            THROW("Cannot load vx_rpp extension (vx_rpp), vxLoadKernels failed " + TOSTR(status))
        else
            LOG("vx_rpp module loaded successfully")
#ifdef ROCAL_VIDEO
        // loading video decoder modules
        if ((status = vxLoadKernels(_context, "vx_amd_media")) != VX_SUCCESS)
            WRN("Cannot load vx_amd_media extension, video decode functionality will not be available")
        else
            LOG("vx_amd_media module loaded")
#endif
        if(_affinity == RocalAffinity::GPU) {
#if ENABLE_HIP
            _device.init_hip(_context);
#else
            _device.init_ocl(_context);
#endif
        }
    }
    catch(const std::exception& e)
    {
        release();
        throw;
    }
}

MasterGraph::Status
MasterGraph::run()
{
    if(!_processing)// The user should not call the run function before the build() is called or while reset() is happening
        return MasterGraph::Status::NOT_RUNNING;

    if(no_more_processed_data()) {
        return MasterGraph::Status::NO_MORE_DATA;
    }
    std::cerr<<"\n _output_tensors.size() :: "<<_output_tensors.size();
    // if(_output_tensors.size() != 0)
    // {
        // std::cerr<<"\n Comes to output tensor size check";
        _ring_buffer.block_if_empty();// wait here if the user thread (caller of this function) is faster in consuming the processed images compare to th output routine in producing them
    // }

    if(_first_run)
    {
        // calling run pops the processed images that have been used by user, when user calls run() for the first time
        // they've not used anything yet, so we don't pop a batch from the _ring_buffer
        _first_run = false;
    }
    else
    {
        // if(_output_tensors.size() != 0)
        // {
            _ring_buffer.pop();
        // }
    }

    // If the last batch of processed imaged has been just popped from the ring_buffer it means user has previously consumed all the processed images.
    // User should check using the IsEmpty() API and not call run() or copy() API when there is no more data. run() will return MasterGraph::Status::NO_MORE_DATA flag to notify it.
    if(no_more_processed_data()) {
        return MasterGraph::Status::NO_MORE_DATA;
    }

    decrease_image_count();

    return MasterGraph::Status::OK;
}

void
MasterGraph::decrease_image_count()
{
    if(!_loop)
        _remaining_images_count -= _user_batch_size;
}
void
MasterGraph::create_single_graph()
{
    // Actual graph creating and calls into adding nodes to graph is deferred and is happening here to enable potential future optimizations
    _graph = std::make_shared<Graph>(_context, _affinity, 0, _gpu_id);
    for(auto& node: _tensor_nodes)
    {
        // Any image not yet created can be created as virtual image
        for(auto& tensor: node->output())
            if(tensor->info().type() == rocALTensorInfo::Type::UNKNOWN)
            {
                tensor->create_virtual(_context, _graph->get());
                _internal_tensors.push_back(tensor);
            }
        node->create(_graph);
    }
    _graph->verify();
}

MasterGraph::Status
MasterGraph::build()
{
    if(_output_tensors.empty())
        THROW("No output images or tensors are there, cannot create the pipeline")

    // Verify all output images have the same dimension, otherwise creating a unified tensor from them is not supported
    _output_tensor_info = _output_tensors.front()->info();
    _max_tensor_type_size = _output_tensor_info.data_type_size();
    for(auto&& output_tensor : _output_tensors)
    {
        rocALTensorInfo tensor_info  = output_tensor->info();
        if(tensor_info.data_type_size() > _max_tensor_type_size)
        {
            _max_tensor_type_size = tensor_info.data_type_size();
            _output_tensor_info = output_tensor->info();
        }
    }

    allocate_output_tensor();
    // if(_output_tensors.size() != 0)
#if ENABLE_HIP
    _ring_buffer.initHip(_mem_type, _device.resources(), tensor_output_byte_size(), _output_tensors.size());
#else
    _ring_buffer.init(_mem_type, _device.resources(), tensor_output_byte_size(), _output_tensors.size());
#endif
    std::cerr<<"\n Initializing Ring buffer with tensor_output_byte_size "<<tensor_output_byte_size();
    create_single_graph();
    start_processing();
    return Status::OK;
}

rocALTensor *
MasterGraph::create_loader_output_tensor(const rocALTensorInfo &info)
{
    /*
    *   NOTE: Output image for a source node needs to be created as a regular (non-virtual) image
    */
    auto output = new rocALTensor(info);
    if(output->create_from_handle(_context) != 0)
        THROW("Creating output image for loader failed");

    _internal_tensors.push_back(output);

    return output;
}

// rocALTensor*
// MasterGraph::create_rocal_loader_output_tensor(const rocALTensorInfo &info)
// {
//     auto output = new rocALTensor(info);
//     if(output->create_from_handle(_context) != 0)
//         THROW("Creating output image for loader failed");

//     _internal_rocal_tensors.push_back(output);

//     return output;
// }

rocALTensor * MasterGraph::create_tensor(const rocALTensorInfo &info, bool is_output)
{
    auto* output = new rocALTensor(info);
    // if the image is not an output image, the image creation is deferred and later it'll be created as a virtual image
    if(is_output)
    {
        if (output->create_from_handle(_context) != 0)
            THROW("Cannot create the tensor from handle")

        _output_tensors.push_back(output);
    }

    return output;
}

// rocALTensor * MasterGraph::create_rocal_tensor(const rocALTensorInfo &info, bool is_output)
// {
//     auto* output = new rocALTensor(info);
//     // if the image is not an output image, the image creation is deferred and later it'll be created as a virtual image
//     if(is_output)
//     {
//         if (output->create_from_handle(_context) != 0)
//             THROW("Cannot create the tensor from handle")

//         _output_rocal_tensors.push_back(output);
//     }

//     return output;
// }


void MasterGraph::release()
{
    LOG("MasterGraph release ...")
    stop_processing();
    _tensor_root_nodes.clear();
    _tensor_map.clear();

    // release all openvx resources.
    vx_status status;
    for(auto& tensor: _output_tensors)
        delete tensor;
    for(auto& tensor: _internal_tensors)
        delete tensor;

    if(_graph != nullptr)
        _graph->release();
    if(_context && (status = vxReleaseContext(&_context)) != VX_SUCCESS)
        LOG ("Failed to call vxReleaseContext " + TOSTR(status))

    _augmented_meta_data = nullptr;
    _meta_data_graph = nullptr;
    _meta_data_reader = nullptr;
    deallocate_output_tensor();
}

MasterGraph::Status
MasterGraph::update_node_parameters()
{
    // Randomize random parameters
    ParameterFactory::instance()->renew_parameters();

    // Apply renewed parameters to VX parameters used in augmentation
    for(auto node:_tensor_nodes)
        node->update_parameters();

    return Status::OK;
}

size_t
MasterGraph::augmentation_branch_count()
{
    return _output_tensors.size();

}

// This is being used only in copy out tensor which will be changed.
RocalColorFormat
MasterGraph::output_color_format()
{
    return _output_tensor_info.color_format();
}

size_t
MasterGraph::tensor_output_width()
{
    return _output_tensor_info.max_width();
}


size_t
MasterGraph::tensor_output_height()
{
    return (_output_tensor_info.max_height() * _internal_batch_size * _user_to_internal_batch_ratio);
}


MasterGraph::Status
MasterGraph::allocate_output_tensor()
{
    // creating a float buffer that can accommodates all output images
    size_t total_size =  _output_tensors.size();
    size_t output_size;
    output_size = tensor_output_byte_size();

    size_t output_float_buffer_size = output_size * total_size;
#if !ENABLE_HIP
    if(processing_on_device_ocl())
    {
        cl_int ret = CL_SUCCESS;
        _output_tensor = nullptr;
        size_t size = output_float_buffer_size*sizeof(cl_float);
        cl_mem clImgFloat  = clCreateBuffer(_device.resources().context,
                                            CL_MEM_READ_WRITE,
                                            size,
                                            nullptr, &ret);

        if (!clImgFloat || ret != CL_SUCCESS)
            THROW("clCreateBuffer of size " + TOSTR(size) + " failed " + TOSTR(ret))

        _output_tensor = clImgFloat;
    }
#else
    if (processing_on_device_hip())
    {
        void *hipMemory = nullptr;
        size_t size = (_out_data_type==RocalTensorDataType::FP32)? output_float_buffer_size*sizeof(float): output_float_buffer_size*sizeof(half);
        hipError_t status = hipMalloc( &hipMemory, size);
        if (status != hipSuccess)
            THROW("ROCAL::hipMalloc of size " + TOSTR(size) + " failed " + TOSTR(status))

        _output_tensor = hipMemory;
    }
#endif
    return Status::OK;
}

MasterGraph::Status
MasterGraph::deallocate_output_tensor()
{
#if !ENABLE_HIP
    if(processing_on_device_ocl() && _output_tensor != nullptr)
        clReleaseMemObject((cl_mem)_output_tensor );
#else
    if(processing_on_device_hip() && _output_tensor != nullptr) {
        hipError_t err = hipFree(_output_tensor );
        if (err != hipSuccess) {
            THROW("MasterGraph::deallocate_output_tensor  hipFree failed " + TOSTR(err))
        }
        _output_tensor = nullptr;
    }
#endif

    return Status::OK;
}

MasterGraph::Status
MasterGraph::reset()
{
    // stop the internal processing thread so that the
    _processing = false;
    _ring_buffer.unblock_writer();
    if(_output_thread.joinable())
        _output_thread.join();
    _ring_buffer.reset();
    // clearing meta ring buffer
    // if random_bbox meta reader is used: read again to get different crops
    // if (_randombboxcrop_meta_data_reader != nullptr)
    //     _randombboxcrop_meta_data_reader->release();
    // resetting loader module to start from the beginning of the media and clear it's internal state/buffers
    _loader_module->reset();
    // restart processing of the images
    _first_run = true;
    _output_routine_finished_processing = false;
    start_processing();
    return Status::OK;
}

size_t
MasterGraph::remaining_images_count()
{
    return (_remaining_images_count >= 0) ? _remaining_images_count:0;
}

RocalMemType
MasterGraph::mem_type()
{
    return _mem_type;
}

Timing
MasterGraph::timing()
{
    Timing t = _loader_module->timing();
    t.image_process_time += _process_time.get_timing();
    t.copy_to_output += _convert_time.get_timing();
    return t;
}


MasterGraph::Status
MasterGraph::copy_output(
        void* out_ptr,
        size_t out_size)
{
    if(no_more_processed_data())
        return MasterGraph::Status::NO_MORE_DATA;

    return Status::NOT_IMPLEMENTED;
    _convert_time.start();
    _convert_time.end();
    return Status::OK;
}

#define CHECK_CL_CALL_RET(x) { cl_int ret; ret = x; if( ret != CL_SUCCESS) THROW("ocl call failed "+STR(#x)+" error "+TOSTR(ret)) }


MasterGraph::Status
MasterGraph::copy_output(void *out_ptr)
{
    std::cerr<<"Inside Copy Output";
    if(no_more_processed_data())
        return MasterGraph::Status::NO_MORE_DATA;

    _convert_time.start();
    // Copies to the output context given by the user
    size_t size;
    size = tensor_output_byte_size();
    std::cerr<<"\n tensor output byte size :: "<<size;
    size_t dest_buf_offset = 0;
#if !ENABLE_HIP
    if(processing_on_device_ocl())
    {
        //NOTE: the CL_TRUE flag is only used on the last buffer read call,
        // to avoid unnecessary sequence of synchronizations

        // get_read_buffers() calls block_if_empty() internally and blocks if buffers are empty until a new batch is processed
        auto output_buffers =_ring_buffer.get_read_buffers();
        auto out_image_idx = output_buffers.size();
        for( auto&& output_handle: output_buffers)
        {
            bool sync_flag = (--out_image_idx == 0) ? CL_TRUE : CL_FALSE;
            cl_int status;
            if((status = clEnqueueReadBuffer(_device.resources().cmd_queue,
                                             (cl_mem) output_handle,
                                             sync_flag?(CL_TRUE):CL_FALSE,
                                             0,
                                             size,
                                             (((unsigned char *)out_ptr) + dest_buf_offset),
                                             0 , nullptr, nullptr)) != CL_SUCCESS)
                THROW("clEnqueueReadBuffer failed: " + TOSTR(status))
            dest_buf_offset += size;
        }
    }
#else
    if(processing_on_device_hip())
    {
        //NOTE: the CL_TRUE flag is only used on the last buffer read call,
        // to avoid unnecessary sequence of synchronizations

        // get_read_buffers() calls block_if_empty() internally and blocks if buffers are empty until a new batch is processed
        auto output_buffers =_ring_buffer.get_read_buffers();
        for( auto&& output_handle: output_buffers)
        {
            hipError_t err = hipMemcpyDtoHAsync((void *)(out_ptr+dest_buf_offset), output_handle, size, _device.resources().hip_stream);
            if (err) {
                THROW("hipMemcpyDtoHAsync failed: " + TOSTR(err))
            }
            dest_buf_offset += size;
        }
        // sync to finish copy
        if (hipStreamSynchronize(_device.resources().hip_stream) != hipSuccess)
            THROW("hipStreamSynchronize failed for hipMemcpy ")

    }
#endif
    else
    {
        // get_host_master_read_buffer is blocking if _ring_buffer is empty, and blocks this thread till internal processing thread process a new batch and store in the _ring_buffer
        std::cerr<<"\n Gonna copy buffer of size "<<size * _output_tensors.size()<<" in host";
        memcpy(out_ptr, _ring_buffer.get_host_master_read_buffer(), size * _output_tensors.size());
    }
    _convert_time.end();
    return Status::OK;
}

ImageNameBatch& operator+=(ImageNameBatch& dest, const ImageNameBatch& src)
{
    dest.insert(dest.end(), src.cbegin(), src.cend());
    return dest;
}

void MasterGraph::output_routine()
{
    INFO("Output routine started with "+TOSTR(_remaining_images_count) + " to load");
#if !ENABLE_HIP
    if(processing_on_device_ocl() && _user_to_internal_batch_ratio != 1)
        THROW("Internal failure, in the GPU processing case, user and input batch size must be equal")
#else
    if(processing_on_device_hip() && _user_to_internal_batch_ratio != 1)
        THROW("Internal failure, in the GPU processing case, user and input batch size must be equal")
#endif
    try {
        while (_processing)
        {
            const size_t tensor_each_cycle_size = tensor_output_byte_size()/_user_to_internal_batch_ratio;
            ImageNameBatch full_batch_image_names = {};
            pMetaDataBatch full_batch_meta_data = nullptr;
            pMetaDataBatch augmented_batch_meta_data = nullptr;
            if (_loader_module->remaining_count() < _user_batch_size)
            {
                // If the internal process routine ,output_routine(), has finished processing all the images, and last
                // processed images stored in the _ring_buffer will be consumed by the user when it calls the run() func
                notify_user_thread();
                // the following call is required in case the ring buffer is waiting for more data to be loaded and there is no more data to process.
                if(_output_tensors.size() != 0)
                    _ring_buffer.release_if_empty();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            // When executing on CPU the internal batch count can be smaller than the user batch count
            // In that case the user_batch_size will be an integer multiple of the _internal_batch_size
            // Multiple cycles worth of internal_batch_size images should be processed to complete a full _user_batch_size
            for(unsigned cycle_idx = 0; cycle_idx < _user_to_internal_batch_ratio; cycle_idx++)
            {
                // Swap handles on the input image, so that new image is loaded to be processed
                // std::cerr << "\nBefore load next\n";
                auto load_ret = _loader_module->load_next();
                // std::cerr << "\nAfter load next\n";
                if (load_ret != LoaderModuleStatus::OK)
                    THROW("Loader module failed to load next batch of images, status " + TOSTR(load_ret))
                if (!_processing)
                    break;
                auto this_cycle_names =  _loader_module->get_id();
                auto decode_image_info = _loader_module->get_decode_image_info();
                auto crop_image_info = _loader_module->get_crop_image_info();

                // std::cerr << "\nThis cycle names: " << this_cycle_names.at(0) << "\n";
                if(this_cycle_names.size() != _internal_batch_size)
                    WRN("Internal problem: names count "+ TOSTR(this_cycle_names.size()))
                // meta_data lookup is done before _meta_data_graph->process() is called to have the new meta_data ready for processing
                if (_meta_data_reader)
                    _meta_data_reader->lookup(this_cycle_names);
                full_batch_image_names += this_cycle_names;
                if (!_processing)
                    break;

                // Swap handles on the output images, so that new processed image will be written to the a new buffer
                for (size_t idx = 0; idx < _output_tensors.size(); idx++)
                {
                    // if(_output_tensors.size() != 0)
                    auto tensor_write_buffer = _ring_buffer.get_write_buffers();
                    if(_affinity == RocalAffinity::GPU)
                    {
                        _output_tensors[idx]->swap_handle(tensor_write_buffer[idx]);
                    }
                    else
                    {
                        // std::cerr<<"\n idx::"<<idx<<"\t tensor_each_cycle_size::"<<tensor_each_cycle_size<<"\t cycle_idx"<<cycle_idx;
                        // Have to change float to the equivalent of max size data type
                        if(_output_tensors[idx]->info().data_type() == RocalTensorDataType::FP32)
                        {
                            auto this_cycle_buffer_ptr = (vx_float32 *) tensor_write_buffer[idx] + tensor_each_cycle_size * cycle_idx;
                            _output_tensors[idx]->swap_handle(this_cycle_buffer_ptr);
                        }
                        else if (_output_tensors[idx]->info().data_type() == RocalTensorDataType::FP16)
                        {
                            auto this_cycle_buffer_ptr = (half *) tensor_write_buffer[idx] + tensor_each_cycle_size * cycle_idx;
                            _output_tensors[idx]->swap_handle(this_cycle_buffer_ptr);
                        }
                        else if(_output_tensors[idx]->info().data_type() == RocalTensorDataType::UINT8)
                        {
                            auto this_cycle_buffer_ptr = (vx_uint8 *) tensor_write_buffer[idx] + tensor_each_cycle_size * cycle_idx;
                            _output_tensors[idx]->swap_handle(this_cycle_buffer_ptr);
                        }
                    }
                }
                if (!_processing)
                    break;

                for(auto node: _tensor_nodes)
                {
                    if(node->_is_ssd)
                    {
                        node->set_meta_data(_augmented_meta_data);
                    }
                }

                update_node_parameters();
                if(_augmented_meta_data)
                {
                    if (_meta_data_graph)
                    {
                        if(_is_random_bbox_crop)
                        {
                            _meta_data_graph->update_random_bbox_meta_data(_augmented_meta_data, decode_image_info, crop_image_info);
                        }
                        else
                        {
                            _meta_data_graph->update_meta_data(_augmented_meta_data, decode_image_info);
                        }
                        _meta_data_graph->process(_augmented_meta_data);
                    }
                    if (full_batch_meta_data)
                        full_batch_meta_data->concatenate(_augmented_meta_data);
                    else
                        full_batch_meta_data = _augmented_meta_data->clone();
                }
                _process_time.start();
                _graph->process();
                _process_time.end();
            }
            if(_output_tensors.size() != 0)
            {
                _ring_buffer.set_meta_data(full_batch_image_names, full_batch_meta_data);
                _ring_buffer.push();
            }
        }
    }
    catch (const std::exception &e)
    {
        ERR("Exception thrown in the process routine: " + STR(e.what()) + STR("\n"));
        _processing = false;
        // _ring_buffer.release_all_blocked_calls();
        _ring_buffer.release_all_blocked_calls();
    }
}

void MasterGraph::start_processing()
{
    _processing = true;
    _remaining_images_count = _loader_module->remaining_count();
    _output_thread = std::thread(&MasterGraph::output_routine, this);
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#else
//  Changing thread scheduling policy and it's priority does not help on latest Ubuntu builds
//  and needs tweaking the Linux security settings , can be turned on for experimentation
#if 0
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    auto thread = _output_thread.native_handle();
    auto ret = pthread_setschedparam(thread, SCHED_FIFO, &params);
    if (ret != 0)
        WRN("Unsuccessful in setting thread realtime priority for process thread err = "+STR(std::strerror(ret)))
#endif
#endif
}

void MasterGraph::stop_processing()
{
    _processing = false;
    _ring_buffer.unblock_reader();
    _ring_buffer.unblock_writer();
    if(_output_thread.joinable())
        _output_thread.join();
}

MetaDataBatch * MasterGraph::create_label_reader(const char *source_path, MetaDataReaderType reader_type)
{
    if( _meta_data_reader)
        THROW("A metadata reader has already been created")
    MetaDataConfig config(MetaDataType::Label, reader_type, source_path);
    _meta_data_reader = create_meta_data_reader(config);
    _meta_data_reader->init(config);
    _meta_data_reader->read_all(source_path);
    if (_augmented_meta_data)
        THROW("Metadata can only have a single output")
    else
        _augmented_meta_data = _meta_data_reader->get_output();
    return _meta_data_reader->get_output();
}

const std::pair<ImageNameBatch,pMetaDataBatch>& MasterGraph::meta_data()
{
    if(_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    return _ring_buffer.get_meta_data();

}

const std::pair<ImageNameBatch,pMetaDataBatch>& MasterGraph::tensor_meta_data()
{
    if(_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    return _ring_buffer.get_meta_data();
}

size_t MasterGraph::compute_optimum_internal_batch_size(size_t user_batch_size, RocalAffinity affinity)
{
    const unsigned MINIMUM_CPU_THREAD_COUNT = 2;
    const unsigned DEFAULT_SMT_COUNT = 2;


    if(affinity == RocalAffinity::GPU)
        return user_batch_size;

    unsigned THREAD_COUNT = std::thread::hardware_concurrency();
    if(THREAD_COUNT >= MINIMUM_CPU_THREAD_COUNT)
        INFO("Can run " + TOSTR(THREAD_COUNT) + " threads simultaneously on this machine")
    else
    {
        THREAD_COUNT = MINIMUM_CPU_THREAD_COUNT;
        WRN("hardware_concurrency() call failed assuming can run " + TOSTR(THREAD_COUNT) + " threads")
    }
    size_t ret = user_batch_size;
    size_t CORE_COUNT = THREAD_COUNT / DEFAULT_SMT_COUNT;
    if(CORE_COUNT <= 0)
        THROW("Wrong core count detected less than 0")

    // for( size_t i = CORE_COUNT; i <= THREAD_COUNT; i++)
    // {
    //     if(user_batch_size % i == 0)
    //     {
    //         ret = i;
    //         break;
    //     }
    // }
CORE_COUNT =1;// for checking 
    for(size_t i = CORE_COUNT; i > 1; i--)
    {
        if(user_batch_size % i == 0)
        {
            ret = i;
            break;
        }
    }
    INFO("User batch size "+ TOSTR(user_batch_size)+" Internal batch size set to "+ TOSTR(ret))
    return ret;
}

size_t MasterGraph::tensor_output_sample_size()
{
    return _output_tensor_info.data_size();
}


// size_t MasterGraph::output_sample_size()
// {
//     return tensor_output_height() * tensor_output_width() * tensor_output_depth();
// }

size_t MasterGraph::tensor_output_byte_size()
{
    // std::cerr<<"\n _output_tensor_info.data_size():: "<<_output_tensor_info.data_size()<<"\t _max_tensor_type_size:: "<<_max_tensor_type_size;
    return _output_tensor_info.data_size() * _max_tensor_type_size;
}


// size_t MasterGraph::tensor_output_depth() // shobi Need to make this generic to all tensor instead of image
// {
//     return _output_tensor_info.channels();
// }

void MasterGraph::notify_user_thread()
{
    if(_output_routine_finished_processing)
        return;
    LOG("Output routine finished processing all images, no more image to be processed")
    _output_routine_finished_processing = true;
}

bool MasterGraph::no_more_processed_data()
{
    return (_output_routine_finished_processing && _ring_buffer.empty());
}
