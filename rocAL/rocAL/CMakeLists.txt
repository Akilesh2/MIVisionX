# MIT License
#
# Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.0)
project(rali)

set(MIN_CMAKE_VER_REQUIRED 3.10)
set(MIN_CXX_VER_REQUIRED 7.4)
set(BUILD_RALI true)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../amd_openvx/cmake)

if(GPU_SUPPORT)
    find_package(AMDRPP QUIET)
    find_package(TurboJpeg QUIET)
    find_package(Protobuf QUIET)
    find_package(FFmpeg QUIET)
    find_package(OpenCV QUIET)
else()
    SET(BUILD_RALI false)
    message("-- ${Yellow}NOTE: rocAL library requires GPU_SUPPORT=ON and BACKEND=OPENCL/HIP${ColourReset}")
endif()

if( GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH ${ROCM_PATH}/hip CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    list(APPEND CMAKE_MODULE_PATH ${HIP_PATH}/cmake)
    # HSA_PATH
    IF(NOT DEFINED ENV{HSA_PATH})
        SET(HSA_PATH ${ROCM_PATH}/hsa)
    ELSE()
        SET(HSA_PATH $ENV{HSA_PATH})
    ENDIF()
    find_package(HIP QUIET)
    if(HIP_FOUND)
        SET(BUILD_RALI true)
        message(STATUS "Found HIP for rocAL: " ${HIP_VERSION} ${HIP_PATH} )
        if(HIP_COMPILER STREQUAL clang)
            set(hip_library_name amdhip64)
        else()
            message(FATAL_ERROR "unsupported hip compiler")
        endif()
    else()
        SET(BUILD_RALI false)
        message("-- ${Yellow}NOTE: rocAL library requires HIP for BACKEND=HIP, Not Found ${ColourReset}")
    endif()
endif()

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
  find_package(OpenCL QUIET)
  if(NOT OpenCL_FOUND)
      set(BUILD_RALI false)
      message("-- ${Yellow}NOTE: rocAL library requires OpenCL for  BACKEND=OPENCL, Not Found ${ColourReset}")
  endif()
endif()

if(GPU_SUPPORT)
    if(NOT AMDRPP_FOUND)
        set(BUILD_RALI false)
        message("-- ${Yellow}NOTE: rocAL library requires AMD's rpp library, Not Found ${ColourReset}")
    else()
        #find the RPP backend type
        set(RPP_BACKEND_OPENCL_FOUND 0)
        set(RPP_BACKEND_HIP_FOUND 0)
        if(EXISTS ${ROCM_PATH}/rpp/include/rpp_backend.h)
            file(READ ${ROCM_PATH}/rpp/include/rpp_backend.h RPP_BACKEND_FILE)
            string(REGEX MATCH "RPP_BACKEND_OPENCL ([0-9]*)" _ ${RPP_BACKEND_FILE})
            set(RPP_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
            string(REGEX MATCH "RPP_BACKEND_HIP ([0-9]*)" _ ${RPP_BACKEND_FILE})
            set(RPP_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
        else()
            message("-- ${Red}WARNING: ${ROCM_PATH}/rpp/include/rpp_backend.h file Not Found. please run the setup script to install latest RPP package ${ColourReset}")
        endif()

        if ("${BACKEND}" STREQUAL "OPENCL" AND OpenCL_FOUND)
            if (NOT RPP_BACKEND_OPENCL_FOUND)
                set(BUILD_RALI false)
                message("-- ${Red}WARNING: RPP with OpenCL backend Not Found (HINT: run setup script with the OCL backend first!) -- rocAL module excluded.${ColourReset}")
            endif()
        elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
            if (NOT RPP_BACKEND_HIP_FOUND)
                set(BUILD_RALI false)
                message("-- ${Red}WARNING: RPP with the HIP backend Not Found (HINT: run setup script with the HIP backend first!) -- rocAL_rpp module excluded.${ColourReset}")
            endif()
        else()
            set(BUILD_RALI false)
            message("-- ${Red}WARNING: GPU support with OpenCL or HIP Not Found -- rocAL module excluded${ColourReset}")
        endif()
    endif()
    if(NOT TurboJpeg_FOUND)
        set(BUILD_RALI false)
        message("-- ${Yellow}NOTE: rocAL library requires TurboJpeg, Not Found ${ColourReset}")
    endif()
    if(NOT Protobuf_FOUND)
        set(BUILD_RALI false)
        message("-- ${Yellow}NOTE: rocAL library requires Protobuf, Not Found ${ColourReset}")
    endif()
    if(CMAKE_VERSION VERSION_LESS ${MIN_CMAKE_VER_REQUIRED})
        set(BUILD_RALI false)
        message("-- ${Yellow}NOTE: rocAL library requires cmake version greater than or equal to ${MIN_CMAKE_VER_REQUIRED}, currently ${CMAKE_VERSION} is installed${ColourReset}")
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${MIN_CXX_VER_REQUIRED})
        set(BUILD_RALI false)
        message("-- ${Yellow}NOTE: rocAL library requires g++ version greater than or equal to ${MIN_CXX_VER_REQUIRED}, currently ${CMAKE_CXX_COMPILER_VERSION} is installed${ColourReset}")
    endif()
endif()

if(${BUILD_RALI})
    # rocAL uses C++ 17 features
    set(CMAKE_CXX_STANDARD 17)
    message("-- ${Green}ROCm Augmentation Library -- rocAL module added${ColourReset}")
    install(FILES
        ../python/rali.py
        ../python/rali_lib.py
        ../python/rali_common.py
        ../python/rali_image.py
        ../python/rali_parameter.py
        ../python/rali_torch.py
        ../python/rali_image_iterator.py
        DESTINATION ./rali/python
        )
    install(FILES
        include/rali_api.h
        include/rali_api_info.h
        include/rali_api_augmentation.h
        include/rali_api_data_loaders.h
        include/rali_api_types.h
        include/rali_api_data_transfer.h
        include/rali_api_parameters.h
        include/rali_api_meta_data.h
        DESTINATION ./include
        )
    if(FFMPEG_FOUND)
        find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
        find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
        find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h)
        find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
    endif()
    include_directories(
                ${OpenCL_INCLUDE_DIRS}
                ${OpenCL_INCLUDE_DIRS}/Headers
                ${AVUTIL_INCLUDE_DIR}
                ${AVCODEC_INCLUDE_DIR}
                ${SWSCALE_INCLUDE_DIR}
                ${AVFORMAT_INCLUDE_DIR}
                ${CMAKE_CURRENT_BINARY_DIR}
                ${TurboJpeg_INCLUDE_DIRS}
                ${PROTOBUF_INCLUDE_DIRS}
                ${AMDRPP_INCLUDE_DIRS}
                ../../amd_openvx/openvx/include/
                ../../amd_openvx_extensions/amd_rpp/include/
                ../../amd_openvx_extensions/amd_media/include/
                ../third_party/rapidjson/include/
                include
                )
    protobuf_generate_cpp(TF_PROTO_SRCS TF_PROTO_HEADERS proto/example.proto proto/feature.proto)
    protobuf_generate_cpp(CAFFE2_PROTO_SRCS CAFFE2_PROTO_HEADERS proto/caffe2_protos.proto)
    protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HEADERS proto/caffe_protos.proto)
    link_directories(${AMDRPP_LIBRARIES_DIR} ${TurboJpeg_LIBRARIES_DIR} ${PROTOBUF_LIBRARY_DIRS} /usr/local/lib/)

    if("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        add_subdirectory(rocAL_hip)
        link_directories(${HIP_PATH}/lib)
        message("-- ${White}rocAL library is going to be built with HIP support ${ColourReset}")
    endif()

    file(GLOB SOURCES "./source/*.cpp")
    add_library(${PROJECT_NAME} SHARED ${SOURCES} ${TF_PROTO_SRCS} ${TF_PROTO_HEADERS} ${CAFFE_PROTO_HEADERS} ${CAFFE_PROTO_SRCS} ${CAFFE2_PROTO_SRCS} ${CAFFE2_PROTO_HEADERS})

    if("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        include_directories(${HIP_PATH}/include ${HSA_PATH}/include rocAL_hip)
        target_compile_definitions(${PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=1)
    else()
        include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=0)
        message("-- ${White}rocAL library is going to be built with OpenCL support ${ColourReset}")
    endif()

    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_SIMD=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DBG_TIMING=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DBGINFO=0)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DBGLOG=0 WRNLOG=0)
    target_compile_definitions(${PROJECT_NAME} PUBLIC TFRECORD_ENABLE=0)
    message("-- ${White}rocAL built with ENABLE_SIMD")
    if (OpenCV_FOUND)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=1)
        include_directories(${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
        if(${OpenCV_VERSION_MAJOR} EQUAL 4)
            target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=1)
        else()
	        target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=0)
        endif()
    else()
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=0)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -msse4.2 -mavx2 -Wall  -fPIC -pg -pthread -std=gnu++14 -Wno-deprecated-declarations")
    message("-- ${White}rocAL - CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}")

    if(NOT FFMPEG_FOUND)
        message("-- ${Yellow}NOTE: rocAL library is going to be built without video decode functionality ${ColourReset}")
        target_link_libraries(${PROJECT_NAME} -fPIC ${PROTOBUF_LIBRARIES} lmdb boost_system boost_filesystem turbojpeg openvx vx_rpp)
    else()
        message("-- ${White}rocAL library is going to be built with video decode functionality ${ColourReset}")
        target_compile_definitions(${PROJECT_NAME} PUBLIC -DRALI_VIDEO)
        target_link_libraries(${PROJECT_NAME} -fPIC ${PROTOBUF_LIBRARIES} ${FFMPEG_LIBRARIES} lmdb turbojpeg openvx vx_rpp)
    endif()
    if("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        target_link_libraries(${PROJECT_NAME} $<TARGET_OBJECTS:rocAL_hip>)
    endif()

    install(TARGETS rali DESTINATION lib)
else()
    message("-- ${Red}WARNING: rocAL module excluded ${ColourReset}")
endif()
