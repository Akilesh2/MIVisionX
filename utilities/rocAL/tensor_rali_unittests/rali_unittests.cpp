/*
MIT License

Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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

#include <iostream>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <vector>

#include "rali_api.h"

using namespace cv;

#define DISPLAY

using namespace std::chrono;

int test(int test_case, const char *path, const char *outName, int rgb, int gpu, int display, int width, int height);
int main(int argc, const char **argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    printf("Usage: image_augmentation <image-dataset-folder> output_image_name <width> <height> test_case display-on-off gpu=1/cpu=0 rgb=1/grayscale =0  \n");
    if (argc < MIN_ARG_COUNT)
        return -1;

    int argIdx = 0;
    const char *path = argv[++argIdx];
    const char *outName = argv[++argIdx];
    int width = atoi(argv[++argIdx]);
    int height = atoi(argv[++argIdx]);

    bool display = 1; // Display the images
    int rgb = 1;      // process color images
    bool gpu = 1;
    int test_case = 3; // For Rotate

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        display = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    test(test_case, path, outName, rgb, gpu, display, width, height);

    return 0;
}

int test(int test_case, const char *path, const char *outName, int rgb, int gpu, int display, int width, int height)
{
    size_t num_threads = 1;
    int inputBatchSize = 2;
    int decode_max_width = width * 2;
    int decode_max_height = height * 2;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (gpu ? "GPU" : "CPU") << " , " << (rgb ? " Color " : " Grayscale ") << std::endl;

    RaliImageColor color_format = (rgb != 0) ? RaliImageColor::RALI_COLOR_RGB24
                                             : RaliImageColor::RALI_COLOR_U8;

    auto handle = raliCreate(inputBatchSize,
                             gpu ? RaliProcessMode::RALI_PROCESS_GPU : RaliProcessMode::RALI_PROCESS_CPU, 0,
                             1);

    if (raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "Could not create the Rali contex\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>> Creating Rali parameters  <<<<<<<<<<<<<<<<*/

    raliSetSeed(0);

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RaliFloatParam rand_crop_area = raliCreateFloatUniformRand(0.3, 0.5);
    RaliIntParam color_temp_adj = raliCreateIntParameter(-50);

    // Creating a custom random object to set a limited number of values to randomize the rotation angle
    const size_t num_values = 3;
    float values[num_values] = {0, 10, 135};
    double frequencies[num_values] = {1, 5, 5};
    RaliFloatParam rand_angle = raliCreateFloatRand(values, frequencies,
                                                    sizeof(values) / sizeof(values[0]));

    //num_values = 2;
    int new_values[2] = {0, 1};
    double new_freq[2] = {40, 60};
    RaliIntParam rand_mirror = raliCreateIntRand(new_values, new_freq, 2);

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RaliMetaData meta_data = raliCreateLabelReader(handle, path);

    RaliTensor input1;
    RaliTensorLayout tensorLayout = RaliTensorLayout::RALI_NHWC;
    RaliTensorOutputType tensorOutputType = RaliTensorOutputType::RALI_UINT8;

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
#ifdef PARTIAL_DECODE
    {
        input1 = raliFusedJpegCrop(handle, path, color_format, num_threads, false, false);
    }
#else
    if (decode_max_height <= 0 || decode_max_width <= 0)
    {
        input1 = raliJpegFileSource(handle, path, color_format, num_threads, false, true, false);
    }
    else
    {
        input1 = raliJpegFileSource(handle, path, color_format, num_threads, false, true, false,
                                    RALI_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
    }
#endif

    if (raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "JPEG source could not initialize : " << raliGetErrorMessage(handle) << std::endl;
        return -1;
    }

    int resize_w = width;
    int resize_h = height;
    RaliTensor image1, image2;

    switch (test_case)
    {
    case 0:
    {
        std::vector<float> mean{0, 0, 0};
        std::vector<float> sdev{1, 1, 1};
        std::cout << ">>>>>>> Running "
                  << " Crop Mirror Normalize Tensor" << std::endl;
        image1 = raliCropMirrorNormalizeTensor(handle, input1, tensorLayout, tensorOutputType, 3, resize_w, resize_h, 0, 0, 0, mean, sdev, true);
    }
    break;
    case 1:
    {
        std::cout << ">>>>>>> Running "
                  << "raliBrightness" << std::endl;
        image1 = raliBrightness(handle, input1, true);
    }
    break;
    case 2:
    {
        std::cout << ">>>>>>> Running "
                  << "raliGamma" << std::endl;
        image1 = raliGamma(handle, input1, true);
    }
    break;
    default:
        std::cout << "Not a valid option! Exiting!\n";
        return -1;
    }
    // Calling the API to verify and build the augmentation graph
    raliVerify(handle);

    if (raliGetStatus(handle) != RALI_OK)
    {
        std::cout << "Could not verify the augmentation graph " << raliGetErrorMessage(handle);
        return -1;
    }

    std::cout << "\n\nAugmented copies count " << raliGetAugmentationBranchCount(handle) << std::endl;

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = raliGetAugmentationBranchCount(handle) * raliGetOutputHeight(handle);
    int w = raliGetOutputWidth(handle);
    int p = ((color_format == RaliImageColor::RALI_COLOR_RGB24) ? 3 : 1);
    const unsigned number_of_cols = 1; //1920 / w;
    cv::Mat mat_output, mat_input;
    switch (tensorOutputType)
    {
    case RALI_FP32:
    {
        auto cv_color_format = ((color_format == RaliImageColor::RALI_COLOR_RGB24) ? CV_32FC3 : CV_8UC1);
        mat_output = cv::Mat(h, w, cv_color_format);
        mat_input = cv::Mat(h, w, cv_color_format);
    }
    break;
    case RALI_UINT8:
    {
        auto cv_color_format = ((color_format == RaliImageColor::RALI_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
        mat_output = cv::Mat(h, w, cv_color_format);
        mat_input = cv::Mat(h, w, cv_color_format);
    }
    break;
    }

    cv::Mat mat_color;
    int col_counter = 0;
    //cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;
    while (raliGetRemainingImages(handle) >= inputBatchSize)
    {
        // std::cerr<<"\n raliGetRemainingImages:: "<<raliGetRemainingImages(handle)<<"\t inputBatchsize:: "<<inputBatchSize  ;
        // std::cerr<<"\n index "<<index;
        index++;
        if (raliRun(handle) != 0)
        {
            // sleep(2);
            break;
        }

        auto last_colot_temp = raliGetIntValue(color_temp_adj);
        raliUpdateIntParameter(last_colot_temp + 1, color_temp_adj);
        switch (tensorOutputType)
        {
        case RALI_FP32:
            raliCopyToTensorOutput(handle, (float *)mat_input.data, h * w * p);
            break;
        case RALI_UINT8:
            raliCopyToTensorOutput(handle, (unsigned char *)mat_input.data, h * w * p);
            break;
        }
        int label_id[inputBatchSize];
        int image_name_length[inputBatchSize];
        raliGetImageLabels(handle, label_id);
        int img_size = raliGetImageNameLen(handle, image_name_length);
        char img_name[img_size];
        raliGetImageName(handle, img_name);
        std::cerr << "\nPrinting image names of batch: " << img_name;
        if (!display)
            continue;

        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
        if (color_format == RaliImageColor::RALI_COLOR_RGB24)
        {
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            //cv::imshow("output", mat_color);
            cv::imwrite(std::to_string(index) + outName, mat_color, compression_params);
            // cv::waitKey(0);
        }
        else {
            //cv::imshow("output", mat_output);
            cv::imwrite(std::to_string(index) + outName, mat_output, compression_params);
        }
        col_counter = (col_counter + 1) % number_of_cols;
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rali_timing = raliGetTimingInfo(handle);
    std::cout << "Load     time " << rali_timing.load_time << std::endl;
    std::cout << "Decode   time " << rali_timing.decode_time << std::endl;
    std::cout << "Process  time " << rali_timing.process_time << std::endl;
    std::cout << "Transfer time " << rali_timing.transfer_time << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    raliRelease(handle);
    mat_input.release();
    mat_output.release();

    return 0;
}
