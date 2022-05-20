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

#include "rocal_api.h"

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

    int return_val = test(test_case, path, outName, rgb, gpu, display, width, height);
    return 0;
}

int test(int test_case, const char *path, const char *outName, int rgb, int gpu, int display, int width, int height)
{
    size_t num_threads = 1;
    int inputBatchSize = 5;
    int decode_max_width = width * 2;
    int decode_max_height = height * 2;
    // int decode_max_width = 0;
    // int decode_max_height = 0;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (gpu ? "GPU" : "CPU") << " , " << (rgb ? " Color " : " Grayscale ") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24
                                             : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(inputBatchSize,
                             gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                             1);

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    rocalSetSeed(0);

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RocalFloatParam rand_crop_area = rocalCreateFloatUniformRand(0.3, 0.5);
    RocalIntParam color_temp_adj = rocalCreateIntParameter(-50);

    // Creating a custom random object to set a limited number of values to randomize the rotation angle
    const size_t num_values = 3;
    float values[num_values] = {0, 10, 135};
    double frequencies[num_values] = {1, 5, 5};
    RocalFloatParam rand_angle = rocalCreateFloatRand(values, frequencies,
                                                    sizeof(values) / sizeof(values[0]));

    //num_values = 2;
    int new_values[2] = {0, 1};
    double new_freq[2] = {40, 60};
    RocalIntParam rand_mirror = rocalCreateIntRand(new_values, new_freq, 2);

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RocalMetaData meta_data = rocalCreateLabelReader(handle, path);

    RocalTensor input1;
    RocalTensorLayout tensorLayout = RocalTensorLayout::ROCAL_NCHW;
    RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_UINT8;

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
#ifdef PARTIAL_DECODE
    {
        input1 = rocalFusedJpegCrop(handle, path, color_format, num_threads, false, false);
    }
#else
    if (decode_max_height <= 0 || decode_max_width <= 0)
    {
        input1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, false, false);
    }
    else
    {
        input1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, false, false,
                                    ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
    }
#endif

    if (rocalGetStatus(handle) != ROCAL_OK)
    { // shobi check error message
        std::cout << " source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    int resize_w = 300;
    int resize_h = 300;
    RocalTensor image1, image2;

    switch (test_case)
    {
    case 0:
    {
        std::vector<float> mean{0, 0, 0};
        std::vector<float> sdev{1, 1, 1};
        std::cout << ">>>>>>> Running "
                  << " Crop Mirror Normalize Tensor" << std::endl;
        std::cerr<<"In unit_Test batchsize "<< inputBatchSize<<"\n";
        image1 = rocalCropMirrorNormalizeTensor(handle, input1, tensorLayout, tensorOutputType, 3, resize_w, resize_h, 0, 0, 0, mean, sdev, true);

    }
    break;
    case 1:
    {

        std::cout << ">>>>>>> Running "
                  << "rocalBrightness" << std::endl;
        image1 = rocalBrightnessTensor(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }
    case 2:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalGamma" << std::endl;
        image1 = rocalGammaTensor(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }
    
    case 3: 
    {
        std::cout << ">>>>>>> Running "
                  << "rocalExposure" << std::endl;
        image1 = rocalExposureTensor(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }
    case 4: 
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorTwist" << std::endl;
        image1 = rocalColorTwistTensor(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }

    case 5: 
    {
        std::cout << ">>>>>>> Running "
                  << "rocalCrop" << std::endl;
        image1 = rocalCropTensor(handle, input1, tensorLayout, tensorOutputType, 3, resize_w, resize_h, 0, 0, 0,true);
    break;
    }
    case 6: 
    {
        std::cout << ">>>>>>> Running "
                  << "rocalResize" << std::endl;
        image1 = rocalResizeTensor(handle, input1, tensorLayout, tensorOutputType, 3, 100, 100, 0,true);
    break;
    }
    default:
        std::cout << "Not a valid option! Exiting!\n";
        return -1;
    }
    // Calling the API to verify and build the augmentation graph
    rocalVerify(handle);

    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    std::cout << "\n\nAugmented copies count " << rocalGetAugmentationBranchCount(handle) << std::endl;

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle);
    int w = rocalGetOutputWidth(handle);
    int p = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1);
    const unsigned number_of_cols = 1; //1920 / w;
    cv::Mat mat_output, mat_input;
    switch (tensorOutputType)
    {
    case ROCAL_FP32:
    {
        auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_32FC3 : CV_8UC1);
        mat_output = cv::Mat(h, w, cv_color_format);
        mat_input = cv::Mat(h, w, cv_color_format);
    }
    break;
    case ROCAL_UINT8:
    {
        auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
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
    while (rocalGetRemainingImages(handle) >= inputBatchSize)
    {
        std::cerr<<"\n rocalGetRemainingImages:: "<<rocalGetRemainingImages(handle)<<"\t inputBatchsize:: "<<inputBatchSize  ;
        std::cerr<<"\n index "<<index;
        index++;
        if (rocalRun(handle) != 0)
        {
            // sleep(2);
            std::cerr<<"\n Inside rocAl run\n";
            break;
        }

        auto last_colot_temp = rocalGetIntValue(color_temp_adj);
        rocalUpdateIntParameter(last_colot_temp + 1, color_temp_adj);
        switch (tensorOutputType)
        {
        case ROCAL_FP32:

            rocalCopyToTensorOutput(handle, (float *)mat_input.data, h * w * p);
            break;
        case ROCAL_UINT8:
            rocalCopyToTensorOutput(handle, (unsigned char *)mat_input.data, h * w * p);
            break;
        }
        int label_id[inputBatchSize];
        int image_name_length[inputBatchSize];
        rocalGetImageLabels(handle, label_id);
        int img_size = rocalGetImageNameLen(handle, image_name_length);
        char img_name[img_size];
        rocalGetImageName(handle, img_name);
        std::cerr << "\nPrinting image names of batch: " << img_name;
        if (!display)
            continue;

        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
        std::string out_filename = std::string(outName) + ".png";   // in case the user specifies non png filename
        if (display)
            out_filename = std::string(outName) + std::to_string(index) + ".png";   // in case the user specifies non png filename

        if (color_format == RocalImageColor::ROCAL_COLOR_RGB24)
        {
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            cv::imwrite(out_filename, mat_color, compression_params);
        }
        else
        {
            cv::imwrite(out_filename, mat_output, compression_params);
        }
        col_counter = (col_counter + 1) % number_of_cols;
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    rocalRelease(handle);
    mat_input.release();
    mat_output.release();
    return 0;
}
