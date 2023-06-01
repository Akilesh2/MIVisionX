#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include "api/rocal_api_types.h"
#include "rocal_api.h"
#include "tensor.h"
#include "api/rocal_api_parameters.h"
#include "api/rocal_api_data_loaders.h"
#include "api/rocal_api_augmentation.h"
#include "api/rocal_api_data_transfer.h"
#include "api/rocal_api_info.h"
namespace py = pybind11;

using float16 = half_float::half;
static_assert(sizeof(float16) == 2, "Bad size");
namespace pybind11
{
    namespace detail
    {
        constexpr int NPY_FLOAT16 = 23;
        // Kinda following: https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
        template <>
        struct npy_format_descriptor<float16>
        {
            static constexpr auto name = _("float16");
            static pybind11::dtype dtype()
            {
                handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
                return reinterpret_borrow<pybind11::dtype>(ptr);
            }
        };
    }
} // namespace pybind11::detail
namespace rocal
{
    using namespace pybind11::literals; // NOLINT
    // PYBIND11_MODULE(rocal_backend_impl, m) {
    static void *ctypes_void_ptr(const py::object &object)
    {
        auto ptr_as_int = getattr(object, "value", py::none());
        if (ptr_as_int.is_none())
        {
            return nullptr;
        }
        void *ptr = PyLong_AsVoidPtr(ptr_as_int.ptr());
        return ptr;
    }

    py::object wrapper(RocalContext context, py::array_t<unsigned char> array)
    {
        auto buf = array.request();
        unsigned char *ptr = (unsigned char *)buf.ptr;
        // call pure C++ function
        int status = rocalCopyToOutput(context, ptr, buf.size);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_image_name_length(RocalContext context, py::array_t<int> array)
    {
        auto buf = array.request();
        int *ptr = (int *)buf.ptr;
        // call pure C++ function
        int length = rocalGetImageNameLen(context, ptr);
        return py::cast(length);
    }

    py::object wrapper_image_name(RocalContext context, int array_len)
    {
        py::array_t<char> array;
        auto buf = array.request();
        char *ptr = (char *)buf.ptr;
        ptr = (char *)calloc(array_len, sizeof(char));
        // call pure C++ function
        rocalGetImageName(context, ptr);
        std::string s(ptr);
        free(ptr);
        return py::bytes(s);
    }

    PYBIND11_MODULE(rocal_pybind, m)
    {
        m.doc() = "Python bindings for the C++ portions of ROCAL";
        // rocal_api.h
        m.def("rocalCreate", &rocalCreate, "Creates context with the arguments sent and returns it",
              py::return_value_policy::reference,
              py::arg("batch_size"),
              py::arg("affinity"),
              py::arg("gpu_id") = 0,
              py::arg("cpu_thread_count") = 1,
              py::arg("prefetch_queue_depth") = 3,
              py::arg("output_data_type") = 0);
        m.def("rocalVerify", &rocalVerify);
        m.def("rocalRun", &rocalRun);
        m.def("rocalRelease", &rocalRelease);
        // rocal_api_types.h
        py::class_<TimingInfo>(m, "TimingInfo")
            .def_readwrite("load_time",&TimingInfo::load_time)
            .def_readwrite("decode_time",&TimingInfo::decode_time)
            .def_readwrite("process_time",&TimingInfo::process_time)
            .def_readwrite("transfer_time",&TimingInfo::transfer_time)
            .def_readwrite("output_routine_time",&TimingInfo::output_routine_time);
        py::class_<rocalTensor>(m, "rocalTensor")
                .def(
                "batch_height",
                [](rocalTensor &output_tensor)
                {
                    return output_tensor.info().max_shape().at(1);
                },
                R"code(
                Returns a tensor buffer's height.
                )code"
            )
            .def(
                "batch_width",
                [](rocalTensor &output_tensor)
                {
                    return output_tensor.info().max_shape().at(0);
                },
                R"code(
                Returns a tensor buffer's width.
                )code"
            )
            .def(
                "batch_size",
                [](rocalTensor &output_tensor)
                {
                    return output_tensor.info().dims().at(0);
                },
                R"code(
                Returns a tensor batch size.
                )code"
            )
            .def(
                "color_format",
                [](rocalTensor &output_tensor)
                {
                    if ((output_tensor.info().color_format() == RocalColorFormat::RGB24) || (output_tensor.info().color_format() == RocalColorFormat::BGR24))
                        return 3;
                    else
                        return 1;
                },
                R"code(
                Returns a tensor batch size.
                )code"
            )
            .def(
            "copy_data", [](rocalTensor &output_tensor, py::object p)
            {
            auto ptr = ctypes_void_ptr(p);
            output_tensor.copy_data(ptr);
            }
            )
            .def(
                "at",
                [](rocalTensor &output_tensor, uint idx)
                {
                    uint h = output_tensor.info().max_shape().at(1);
                    uint w = output_tensor.info().max_shape().at(0);

                    if (output_tensor.info().layout() == RocalTensorlayout::NHWC)
                    {
                        unsigned c = output_tensor.info().dims().at(3);
                        return py::array(py::buffer_info(
                            ((unsigned char *)(output_tensor.buffer())) + idx * c * h * w,
                            sizeof(unsigned char),
                            py::format_descriptor<unsigned char>::format(),
                            output_tensor.info().num_of_dims() - 1,
                            {h, w, c},
                            {sizeof(unsigned char) * w * c, sizeof(unsigned char) * c, sizeof(unsigned char)}));
                    }

                    else if (output_tensor.info().layout() == RocalTensorlayout::NCHW)
                    {
                        unsigned n = output_tensor.info().dims().at(0);
                        unsigned c = output_tensor.info().dims().at(1);
                        return py::array(py::buffer_info(
                            ((unsigned char *)(output_tensor.buffer())) + idx * c * h * w,
                            sizeof(unsigned char),
                            py::format_descriptor<unsigned char>::format(),
                            output_tensor.info().num_of_dims(),
                            {c, h, w},
                            {sizeof(unsigned char) * c * h * w, sizeof(unsigned char) * h * w, sizeof(unsigned char) * w, sizeof(unsigned char)}));
                    }
                },
                "idx"_a,
                R"code(
                Returns a rocAL tensor at given position `i` in the rocalTensorlist.
                )code",
                py::keep_alive<0, 1>());

        // .def_readwrite("swap_handle",&rocalTensor::swap_handle);
        py::class_<rocalTensorList>(m, "rocalTensorList")
            .def(
                "__getitem__",
                [](rocalTensorList &output_tensor_list, uint idx)
                {
                    return output_tensor_list.at(idx);
                },
                R"code(
                Returns a tensor at given position in the list.
                )code")

            .def("at",
                [](rocalTensorList &output_tensor_list, uint idx)
                {
                    uint h = output_tensor_list.at(idx)->info().max_shape().at(1);
                    uint w = output_tensor_list.at(idx)->info().max_shape().at(0);

                    if (output_tensor_list.at(idx)->info().layout() == RocalTensorlayout::NHWC)
                    {
                        unsigned n = output_tensor_list.at(idx)->info().dims().at(0);
                        unsigned c = output_tensor_list.at(idx)->info().dims().at(3);
                        return py::array(py::buffer_info(
                            (unsigned char *)(output_tensor_list.at(idx)->buffer()),
                            sizeof(unsigned char),
                            py::format_descriptor<unsigned char>::format(),
                            output_tensor_list.at(idx)->info().num_of_dims(),
                            {n, h, w, c},
                            {sizeof(unsigned char) * w * h * c, sizeof(unsigned char) * w * c, sizeof(unsigned char) * c, sizeof(unsigned char)}));
                    }

                    else if (output_tensor_list.at(idx)->info().layout() == RocalTensorlayout::NCHW)
                    {
                        unsigned n = output_tensor_list.at(idx)->info().dims().at(0);
                        unsigned c = output_tensor_list.at(idx)->info().dims().at(1);
                        return py::array(py::buffer_info(
                            (unsigned char *)(output_tensor_list.at(idx)->buffer()),
                            sizeof(unsigned char),
                            py::format_descriptor<unsigned char>::format(),
                            output_tensor_list.at(idx)->info().num_of_dims(),
                            {n, c, h, w},
                            {sizeof(unsigned char) * c * h * w, sizeof(unsigned char) * h * w, sizeof(unsigned char) * w, sizeof(unsigned char)}));
                    }
                },
                "idx"_a,
                R"code(
                Returns a rocAL tensor at given position `i` in the rocalTensorlist.
                )code",
                py::keep_alive<0, 1>());
        py::class_<rocalTensorInfo>(m, "rocalTensorInfo");

        py::module types_m = m.def_submodule("types");
        types_m.doc() = "Datatypes and options used by ROCAL";
        py::enum_<RocalStatus>(types_m, "RocalStatus", "Status info")
            .value("OK", ROCAL_OK)
            .value("CONTEXT_INVALID", ROCAL_CONTEXT_INVALID)
            .value("RUNTIME_ERROR", ROCAL_RUNTIME_ERROR)
            .value("UPDATE_PARAMETER_FAILED", ROCAL_UPDATE_PARAMETER_FAILED)
            .value("INVALID_PARAMETER_TYPE", ROCAL_INVALID_PARAMETER_TYPE)
            .export_values();
        py::enum_<RocalProcessMode>(types_m, "RocalProcessMode", "Processing mode")
            .value("GPU", ROCAL_PROCESS_GPU)
            .value("CPU", ROCAL_PROCESS_CPU)
            .export_values();
        py::enum_<RocalTensorOutputType>(types_m, "RocalTensorOutputType", "Tensor types")
            .value("FLOAT", ROCAL_FP32)
            .value("FLOAT16", ROCAL_FP16)
            .value("UINT8", ROCAL_UINT8)
            .export_values();
        py::enum_<RocalImageSizeEvaluationPolicy>(types_m, "RocalImageSizeEvaluationPolicy", "Decode size policies")
            .value("MAX_SIZE", ROCAL_USE_MAX_SIZE)
            .value("USER_GIVEN_SIZE", ROCAL_USE_USER_GIVEN_SIZE)
            .value("MOST_FREQUENT_SIZE", ROCAL_USE_MOST_FREQUENT_SIZE)
            .value("MAX_SIZE_ORIG", ROCAL_USE_MAX_SIZE_RESTRICTED)
            .value("USER_GIVEN_SIZE_ORIG", ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED)
            .export_values();
        py::enum_<RocalImageColor>(types_m, "RocalImageColor", "Image type")
            .value("RGB", ROCAL_COLOR_RGB24)
            .value("BGR", ROCAL_COLOR_BGR24)
            .value("GRAY", ROCAL_COLOR_U8)
            .value("RGB_PLANAR", ROCAL_COLOR_RGB_PLANAR)
            .export_values();
        py::enum_<RocalTensorLayout>(types_m, "RocalTensorLayout", "Tensor layout type")
            .value("NHWC", ROCAL_NHWC)
            .value("NCHW", ROCAL_NCHW)
            .export_values();
        py::enum_<RocalDecodeDevice>(types_m, "RocalDecodeDevice", "Decode device type")
            .value("HARDWARE_DECODE", ROCAL_HW_DECODE)
            .value("SOFTWARE_DECODE", ROCAL_SW_DECODE)
            .export_values();
        py::enum_<RocalDecoderType>(types_m,"RocalDecoderType", "Rocal Decoder Type")
            .value("DECODER_TJPEG",ROCAL_DECODER_TJPEG)
            .value("DECODER_OPENCV",ROCAL_DECODER_OPENCV)
            .value("DECODER_HW_JEPG",ROCAL_DECODER_HW_JEPG)
            .value("DECODER_VIDEO_FFMPEG_SW",ROCAL_DECODER_VIDEO_FFMPEG_SW)
            .value("DECODER_VIDEO_FFMPEG_HW",ROCAL_DECODER_VIDEO_FFMPEG_HW)
            .export_values();
        py::enum_<RocalResizeScalingMode>(types_m,"RocalResizeScalingMode","Decode size policies")
            .value("SCALING_MODE_DEFAULT",ROCAL_SCALING_MODE_DEFAULT)
            .value("SCALING_MODE_STRETCH",ROCAL_SCALING_MODE_STRETCH)
            .value("SCALING_MODE_NOT_SMALLER",ROCAL_SCALING_MODE_NOT_SMALLER)
            .value("SCALING_MODE_NOT_LARGER",ROCAL_SCALING_MODE_NOT_LARGER)
            .export_values();
        py::enum_<RocalResizeInterpolationType>(types_m,"RocalResizeInterpolationType","Decode size policies")
            .value("NEAREST_NEIGHBOR_INTERPOLATION",ROCAL_NEAREST_NEIGHBOR_INTERPOLATION)
            .value("LINEAR_INTERPOLATION",ROCAL_LINEAR_INTERPOLATION)
            .value("CUBIC_INTERPOLATION",ROCAL_CUBIC_INTERPOLATION)
            .value("LANCZOS_INTERPOLATION",ROCAL_LANCZOS_INTERPOLATION)
            .value("GAUSSIAN_INTERPOLATION",ROCAL_GAUSSIAN_INTERPOLATION)
            .value("TRIANGULAR_INTERPOLATION",ROCAL_TRIANGULAR_INTERPOLATION)
            .export_values();
        // rocal_api_info.h
        m.def("getRemainingImages", &rocalGetRemainingImages);
        m.def("isEmpty", &rocalIsEmpty);
        m.def("getStatus", rocalGetStatus);
        m.def("rocalGetErrorMessage", &rocalGetErrorMessage);
        m.def("rocalGetTimingInfo", &rocalGetTimingInfo);
        m.def("getTimingInfo", &rocalGetTimingInfo);
        m.def("setOutputImages", &rocalSetOutputs);
        m.def("labelReader", &rocalCreateLabelReader, py::return_value_policy::reference);
        m.def("COCOReader", &rocalCreateCOCOReader, py::return_value_policy::reference);
        // rocal_api_meta_data.h
        m.def("RandomBBoxCrop", &rocalRandomBBoxCrop);
        m.def("BoxEncoder",&rocalBoxEncoder);
        m.def("getImageId", [](RocalContext context, py::array_t<int> array)
        {
            auto buf = array.request();
            int* ptr = (int*) buf.ptr;
            return rocalGetImageId(context,ptr);
        }
        );
        m.def("getImgSizes", [](RocalContext context, py::array_t<int> array)
        {
            auto buf = array.request();
            int* ptr = (int*) buf.ptr;
            // call pure C++ function
            rocalGetImageSizes(context,ptr);
        }
        );
        // rocal_api_parameter.h
        m.def("setSeed", &rocalSetSeed);
        m.def("getSeed", &rocalGetSeed);
        m.def("CreateIntUniformRand", &rocalCreateIntUniformRand);
        m.def("CreateFloatUniformRand", &rocalCreateFloatUniformRand);
        m.def("CreateIntRand", [](std::vector<int> values, std::vector<double> frequencies)
              { return rocalCreateIntRand(values.data(), frequencies.data(), values.size()); });
        m.def("CreateFloatRand", &rocalCreateFloatRand);
        m.def("CreateIntParameter", &rocalCreateIntParameter);
        m.def("CreateFloatParameter", &rocalCreateFloatParameter);
        m.def("UpdateIntParameter", &rocalUpdateIntParameter);
        m.def("UpdateFloatParameter", &rocalUpdateFloatParameter);
        m.def("GetIntValue", &rocalGetIntValue);
        m.def("GetFloatValue", &rocalGetFloatValue);
        m.def("SetRandomPixelMaskConfig",&rocalSetRandomPixelMaskConfig);
        // rocal_api_data_transfer.h
        // m.def("rocalGetOutputTensors",&rocalGetOutputTensors, return_value_policy::reference);
        m.def("rocalGetOutputTensors", [](RocalContext context)
              {
            rocalTensorList * tl = rocalGetOutputTensors(context);
            py::list list;
            unsigned int size_of_tensor_list = tl->size();
            for (uint i =0; i< size_of_tensor_list; i++)
                list.append(tl->at(i));
            return list; });
        m.def(
            "rocalGetImageLabels", [](RocalContext context)
    {
            rocalTensorList *labels = rocalGetImageLabels(context);
            // std::cerr<<"LABELS SIZE ::"<<labels->size();
            // for (int i = 0; i < labels->size(); i++) {
            //     int *labels_buffer = (int *)(labels->at(i)->buffer());
            //     std::cerr << ">>>>> LABELS : " << labels_buffer[0] << "\t";
            // }
            return py::array(py::buffer_info(
                            (int *)(labels->at(0)->buffer()),
                            sizeof(int),
                            py::format_descriptor<int>::format(),
                            1,
                            {labels->size()},
                            {sizeof(int) }));
    }
            );
            m.def(
            "rocalGetBoundingBoxLabel", [](RocalContext context)
    {
            rocalTensorList *labels = rocalGetBoundingBoxLabel(context);
            return py::array(py::buffer_info(
                            (int *)(labels->at(0)->buffer()),
                            sizeof(int),
                            py::format_descriptor<int>::format(),
                            1,
                            {labels->size()},
                            {sizeof(int) }));
    }
            );
            m.def(
            "rocalGetBoundingBoxCords", [](RocalContext context)
    {
            py::list list;
            rocalTensorList *bbox_coords = rocalGetBoundingBoxCords(context);
            for(int i = 0; i < bbox_coords->size(); i++)
            {
                py::list single_image;
                py::tuple bbox_coord;
                float *bbox_buffer = (float *)(bbox_coords->at(i)->buffer());
                for(int j = 0, j4 = 0; j < bbox_coords->at(i)->info().dims().at(0); j++, j4 = j * 4)
                {
                    bbox_coord = py::make_tuple(bbox_buffer[j4], bbox_buffer[j4+1], bbox_buffer[j4+2], bbox_buffer[j4+3]);
                    single_image.append(bbox_coord);
                }
                list.append(single_image);
            }
            return list;
    }
            );
        m.def(
            "rocalGetPixelwiseLabels", [](RocalContext context)
    {
            rocalTensorList * bbox_labels = rocalGetBoundingBoxLabel(context);
            rocalTensorList *labels = rocalGetPixelwiseLabels(context);
            py::list labels_array_list;
            for(int i =0; i < bbox_labels->size(); i++)
            {
                py::array_t<int> labels_array = py::array(py::buffer_info(
                                (int *)(labels->at(i)->buffer()),
                                sizeof(int),
                                py::format_descriptor<int>::format(),
                                1,
                                {labels->at(i)->info().dims().at(0) * labels->at(i)->info().dims().at(1)},
                                {sizeof(int) }));
                labels_array_list.append(labels_array);
            }
            return labels_array_list;
    }
            );
        m.def(
            "rocalRandomMaskPixel", [](RocalContext context)
    {
            rocalTensorList * bbox_labels = rocalGetBoundingBoxLabel(context);
            rocalTensorList * random_mask_pixel = rocalRandomMaskPixel(context);
            py::list random_mask_pixel_array_list;
            
            for(int i =0; i < bbox_labels->size(); i++)
            {
                py::list centre_coordinate_list;
                unsigned int* random_mask_image = (unsigned int *)(random_mask_pixel->at(i)->buffer());
                centre_coordinate_list.append(random_mask_image[0]);
                centre_coordinate_list.append(random_mask_image[1]);
                random_mask_pixel_array_list.append(centre_coordinate_list);
            }
            return random_mask_pixel_array_list;
    }
            );
        m.def(
            "rocalGetBoundingBoxCount", [](RocalContext context)
    {
            unsigned total_number_of_objects_per_batch = rocalGetBoundingBoxCount(context);
            return total_number_of_objects_per_batch;
    }
            );
        m.def(
            "rocalGetMaskCount", [](RocalContext context, py::array_t<int> array)
    {
        auto buf = array.mutable_data();
        std::cerr << "Comes to the PYBIND CALL";
        unsigned count = rocalGetMaskCount(context, buf); //total number of polygons in complete batch
        return count;
    }
            );
        m.def(
            "rocalGetMaskCoordinates", [](RocalContext context, py::array_t<int> polygon_size, py::array_t<int> mask_count)
    {
            auto buf = polygon_size.request();
            int* polygon_size_ptr = (int*) buf.ptr;
            // call pure C++ function
            rocalTensorList * mask_data = rocalGetMaskCoordinates(context, polygon_size_ptr);
            rocalTensorList * bbox_labels = rocalGetBoundingBoxLabel(context);
            py::list complete_list;
            int poly_cnt = 0;
            int prev_object_cnt = 0;
            auto mask_count_buf = mask_count.request();
            int* mask_count_ptr = (int*) mask_count_buf.ptr;
            for(int i = 0; i < bbox_labels->size(); i++) // nbatchSize
            {
                float *mask_buffer = (float *)(mask_data->at(i)->buffer());
                // std::cout << "\n>>>>>>> MASK COORDS : " << bbox_labels->at(i)->info().dims().at(0) << std::endl;
                py::list single_image;
                unsigned int mask_idx = 0;
                for(unsigned j = prev_object_cnt; j < bbox_labels->at(i)->info().dims().at(0) + prev_object_cnt; j++)
                {
                    // std::cout << "Mask idx : " << j << "Polygons : " << mask_count[j] << " Count : " << polygon_size_ptr[poly_cnt] << "[" ;
                    py::list polygons_buffer;
                    for(int k = 0; k < mask_count_ptr[j]; k++)
                    {
                        py::list single_mask;
                        single_mask.append(mask_idx);
                        mask_idx++;
                        py::list mask_buffer_coordinates;
                        // std::cout << "[";
                        for(int l = 0; l < polygon_size_ptr[poly_cnt]; l++)
                        {
                            // std::cout << mask_buffer[l] << ", ";
                            single_image.append(mask_buffer[l]);
                        }
                        // std::cout << "]";
                        mask_buffer += polygon_size_ptr[poly_cnt++];
                    }
                    // std::cout << "]\n";
                }
                prev_object_cnt += bbox_labels->at(i)->info().dims().at(0);
                complete_list.append(single_image);
            }
            return complete_list;
    }
            );
        m.def(
            "rocalSelectMask", [](RocalContext context, std::vector<int> mask_ids)
    {
            rocalTensorList * bbox_labels = rocalGetBoundingBoxLabel(context);
            rocalTensorList * bbox_coords = rocalGetBoundingBoxCords(context);

            std::vector<std::vector<int>> sel_vertices_counts;
            std::vector<std::vector<int>> sel_mask_ids;
            // std::vector<int> mask_ids{0, 1}; // Should this be passed by the user ?
            rocalTensorList * select_mask_polygon = rocalSelectMask(context,
                                                                      mask_ids,
                                                                      sel_vertices_counts,
                                                                      sel_mask_ids,
                                                                      false);
            
            py::list per_image_select_mask;
            for (int i = 0; i < bbox_labels->size(); i++) {
                    // std::cout << "\n>>>>>>> Selected polygons and vertices : " << std::endl;
                    float* select_mask_polygon_buffer = (float *)(select_mask_polygon->at(i)->buffer());
                    auto sel_vertices_count_per_image = sel_vertices_counts[i];
                    auto sel_mask_ids_per_image = sel_mask_ids[i];
                    int cnt = 0;
                    py::dict mask_select_polygon_dict;
                    for (int j = 0; j < mask_ids.size(); j++) {
                        // std::cout << "Mask id: " << mask_ids[j] << "[";
                        py::list select_mask_polygon_list;
                        for (int k = 0; k < sel_vertices_count_per_image[j]; k++) {
                            // std::cout << select_mask_polygon_buffer[cnt++] << " ,";
                            select_mask_polygon_list.append(select_mask_polygon_buffer[cnt++]);
                        }
                        // std::cout << "]" << std::endl;
                        std::string key = std::to_string(mask_ids[j]);
                        mask_select_polygon_dict[py::str(key)] = select_mask_polygon_list;
                    }
                per_image_select_mask.append(mask_select_polygon_dict);
                }
            return per_image_select_mask;
    }
            );

        m.def(
            "rocalGetEncodedBoxesAndLables", [](RocalContext context,uint batch_size, uint num_anchors)
            {
                auto vec_pair_labels_boxes = rocalGetEncodedBoxesAndLables(context, batch_size * num_anchors);
                auto labels_buf_ptr = (int*)(vec_pair_labels_boxes[0]->at(0)->buffer());
                auto bboxes_buf_ptr = (float*)(vec_pair_labels_boxes[1]->at(0)->buffer());

                py::array_t<int> labels_array = py::array_t<int>(py::buffer_info(
                            labels_buf_ptr,
                            sizeof(int),
                            py::format_descriptor<int>::format(),
                            2,
                            {batch_size, num_anchors},
                            {num_anchors*sizeof(int), sizeof(int)}));

                py::array_t<float> bboxes_array = py::array_t<float>(py::buffer_info(
                            bboxes_buf_ptr,
                            sizeof(float),
                            py::format_descriptor<float>::format(),
                            1,
                            {batch_size * num_anchors * 4},
                            {sizeof(float)} ));

        return std::make_pair(labels_array, bboxes_array);
            }
        );
        // rocal_api_data_loaders.h
        m.def("ImageDecoder", &rocalJpegFileSource, "Reads file from the source given and decodes it according to the policy",
              py::return_value_policy::reference,
              py::arg("context"),
              py::arg("source_path"),
              py::arg("color_format"),
              py::arg("num_threads"),
              py::arg("is_output") = false,
              py::arg("shuffle") = false,
              py::arg("loop") = false,
              py::arg("decode_size_policy") = ROCAL_USE_MOST_FREQUENT_SIZE,
              py::arg("max_width") = 0,
              py::arg("max_height") = 0,
              py::arg("dec_type") = 0);
        m.def("ImageDecoderShard", &rocalJpegFileSourceSingleShard, "Reads file from the source given and decodes it according to the shard id and number of shards",
              py::return_value_policy::reference,
              py::arg("context"),
              py::arg("source_path"),
              py::arg("color_format"),
              py::arg("shard_id"),
              py::arg("shard_count"),
              py::arg("is_output") = false,
              py::arg("shuffle") = false,
              py::arg("loop") = false,
              py::arg("decode_size_policy") = ROCAL_USE_MOST_FREQUENT_SIZE,
              py::arg("max_width") = 0,
              py::arg("max_height") = 0,
              py::arg("dec_type") = 0);
        m.def("FusedDecoderCropShard",&rocalFusedJpegCropSingleShard,"Reads file from the source and decodes them partially to output random crops",
            py::return_value_policy::reference);
        m.def("COCO_ImageDecoderShard",&rocalJpegCOCOFileSourceSingleShard,"Reads file from the source given and decodes it according to the shard id and number of shards",
            py::return_value_policy::reference);
        m.def("COCO_ImageDecoderSliceShard",&rocalJpegCOCOFileSourcePartialSingleShard,"Reads file from the source given and decodes it according to the policy",
            py::return_value_policy::reference);
        m.def("Resize",&rocalResize, "Resizes the image ",py::return_value_policy::reference);
        m.def("ColorTwist",&rocalColorTwist, py::return_value_policy::reference);
        m.def("rocalResetLoaders", &rocalResetLoaders);
        // rocal_api_augmentation.h
        m.def("Brightness", &rocalBrightness,
              py::return_value_policy::reference);
        m.def("CropMirrorNormalize",&rocalCropMirrorNormalize, py::return_value_policy::reference);
        // m.def("Crop", &rocalCrop, py::return_value_policy::reference);
        m.def("CenterCropFixed", &rocalCropCenterFixed, py::return_value_policy::reference);

    }
}
