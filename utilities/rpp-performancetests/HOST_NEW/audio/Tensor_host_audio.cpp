#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include "rpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half/half.hpp>
#include <fstream>

// Include this header file to use functions from libsndfile
#include <sndfile.h>

// libsndfile can handle more than 6 channels but we'll restrict it to 6
#define	MAX_CHANNELS 6

using namespace std;
using half_float::half;

typedef half Rpp16f;

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 3;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_host_audio <src folder> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:3>\n");
        return -1;
    }

    char *src = argv[1];
    int ip_bitDepth = atoi(argv[2]);
    int test_case = atoi(argv[3]);

    // Set case names
    char funcName[1000];
    switch (test_case)
    {
        case 0:
            strcpy(funcName, "non_silent_region_detection");
            break;
        case 1:
            strcpy(funcName, "to_decibels");
            break;
        case 2:
            strcpy(funcName, "pre_emphasis_filter");
            break;
        case 3:
            strcpy(funcName, "down_mixing");
            break;
        case 4:
            strcpy(funcName, "slice");
            break;
        case 5:
            strcpy(funcName, "mel_filter_bank");
            break;
        default:
            strcpy(funcName, "test_case");
            break;
    }

     // Initialize tensor descriptors
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    // Set src/dst data types in tensor descriptors
    if (ip_bitDepth == 2)
    {
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    }

    // Other initializations
    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxChannels = 0, maxLength = 0;
    int maxDstLength = 0;
    unsigned long long count = 0;
    unsigned long long iBufferSize = 0, oBufferSize = 0;
    static int noOfAudioFiles = 0;

    // String ops on function name
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");

    char func[1000];
    strcpy(func, funcName);

    // Get number of audio files
    struct dirent *de;
    DIR *dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfAudioFiles += 1;
    }
    closedir(dr);

    // Initialize the AudioPatch for source
    Rpp32s *inputAudioSize = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));
    Rpp32s *srcLengthTensor = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));
    Rpp32s *channelsTensor = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));

    // Set maxLength
    char audioNames[noOfAudioFiles][1000];

    dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        strcpy(audioNames[count], de->d_name);
        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, audioNames[count]);

        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;

        //The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (!(infile = sf_open (temp, SFM_READ, &sfinfo)) || sfinfo.channels > MAX_CHANNELS)
        {
            sf_close (infile);
            continue;
        }

        inputAudioSize[count] = sfinfo.frames * sfinfo.channels;
        srcLengthTensor[count] = sfinfo.frames;
        channelsTensor[count] = sfinfo.channels;
        maxLength = std::max(maxLength, srcLengthTensor[count]);
        maxChannels = std::max(maxChannels, channelsTensor[count]);

        // Close input
        sf_close (infile);
        count++;
    }
    closedir(dr);

    // Set numDims, offset, n/c/h/w values for src/dst
    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    srcDescPtr->offsetInBytes = 0;
    dstDescPtr->offsetInBytes = 0;

    srcDescPtr->n = noOfAudioFiles;
    dstDescPtr->n = noOfAudioFiles;

    srcDescPtr->h = 1;
    dstDescPtr->h = 1;

    srcDescPtr->w = maxLength;
    dstDescPtr->w = maxLength;

    srcDescPtr->c = maxChannels;
    if(test_case == 3)
        dstDescPtr->c = 1;
    else
        dstDescPtr->c = maxChannels;

    // Optionally set w stride as a multiple of 8 for src
    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
    srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
    srcDescPtr->strides.wStride = srcDescPtr->c;
    srcDescPtr->strides.cStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
    dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
    dstDescPtr->strides.wStride = dstDescPtr->c;
    dstDescPtr->strides.cStride = 1;

    // Set buffer sizes for src/dst
    iBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)srcDescPtr->n;
    oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;

    // Initialize host buffers for input & output
    Rpp32f *inputf32 = (Rpp32f *)calloc(iBufferSize, sizeof(Rpp32f));
    Rpp32f *outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));

    i = 0;
    dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        Rpp32f *input_temp_f32;
        input_temp_f32 = inputf32 + (i * srcDescPtr->strides.nStride);

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        strcpy(audioNames[count], de->d_name);
        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, audioNames[count]);

        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;

        // The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (!(infile = sf_open (temp, SFM_READ, &sfinfo)) || sfinfo.channels > MAX_CHANNELS)
        {
            sf_close (infile);
            continue;
        }

        int bufferLength = sfinfo.frames * sfinfo.channels;
        if(ip_bitDepth == 2)
        {
            readcount = (int) sf_read_float (infile, input_temp_f32, bufferLength);
            if(readcount != bufferLength)
                std::cerr<<"F32 Unable to read audio file completely"<<std::endl;
        }
        i++;

        // Close input
        sf_close (infile);
    }
    closedir(dr);

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, noOfAudioFiles);
    printf("\nRunning %s 100 times (each time with a batch size of %d audio files) and computing mean statistics...", func, noOfAudioFiles);

    double max_time_used = 0, min_time_used = 500, avg_time_used = 0;
    string test_case_name;
    for (int perfRunCount = 0; perfRunCount < 100; perfRunCount++)
    {
        clock_t start, end;
        double start_omp, end_omp;
        double cpu_time_used, omp_time_used;

        switch (test_case)
        {
            case 0:
            {
                test_case_name = "non_silent_region_detection";
                Rpp32s detectionIndex[noOfAudioFiles];
                Rpp32s detectionLength[noOfAudioFiles];
                Rpp32f cutOffDB[noOfAudioFiles];
                Rpp32s windowLength[noOfAudioFiles];
                Rpp32f referencePower[noOfAudioFiles];
                Rpp32s resetInterval[noOfAudioFiles];
                bool referenceMax[noOfAudioFiles];

                for (i = 0; i < noOfAudioFiles; i++)
                {
                    cutOffDB[i] = -60.0;
                    windowLength[i] = 3;
                    referencePower[i] = 1.0;
                    resetInterval[i] = -1;
                    referenceMax[i] = true;
                }

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 2)
                {
                    rppt_non_silent_region_detection_host(inputf32, srcDescPtr, inputAudioSize, detectionIndex, detectionLength, cutOffDB, windowLength, referencePower, resetInterval, referenceMax, handle);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case 1:
            {
                test_case_name = "to_decibels";
                Rpp32f cutOffDB = -200.0;
                Rpp32f multiplier = 10.0;
                Rpp32f referenceMagnitude = 0.0;

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 2)
                {
                    rppt_to_decibels_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, cutOffDB, multiplier, referenceMagnitude);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case 2:
            {
                test_case_name = "pre_emphasis_filter";
                Rpp32f coeff[noOfAudioFiles];
                for (i = 0; i < noOfAudioFiles; i++)
                    coeff[i] = 0.97;
                RpptAudioBorderType borderType = RpptAudioBorderType::CLAMP;

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 2)
                {
                    rppt_pre_emphasis_filter_host(inputf32, srcDescPtr, outputf32, dstDescPtr, inputAudioSize, coeff, borderType);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case 3:
            {
                test_case_name = "down_mixing";
                bool normalizeWeights = false;

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 2)
                {
                    rppt_down_mixing_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, channelsTensor, normalizeWeights);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case 4:
            {
                test_case_name = "slice";
                bool normalizedAnchor = false;
                bool normalizedShape = false;
                Rpp32s anchor[noOfAudioFiles];
                Rpp32s shape[noOfAudioFiles];
                Rpp32f fillValues[noOfAudioFiles];
                Rpp32s axes = 0;
                RpptOutOfBoundsPolicy policyType = RpptOutOfBoundsPolicy::ERROR;
                for (i = 0; i < noOfAudioFiles; i++)
                {
                    anchor[i] = 100;
                    shape[i] = 200;
                    fillValues[i] = 0.0f;
                }

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 2)
                {
                    rppt_slice_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, anchor, shape, &axes, fillValues, normalizedAnchor, normalizedShape, policyType);
                }
                else
                    missingFuncFlag = 1;

                break;
            }
            case 5:
            {
                test_case_name = "mel_filter_bank";

                RpptImagePatch *srcDims = (RpptImagePatch *) calloc(noOfAudioFiles, sizeof(RpptImagePatch));
                srcDims[0].width = 225;
                srcDims[0].height = 257;
                Rpp32f sampleRate = 16000;
                Rpp32f minFreq = 0.0;
                Rpp32f maxFreq = sampleRate / 2;
                RpptMelScaleFormula melFormula = RpptMelScaleFormula::SLANEY;
                Rpp32s numFilter = 128;
                bool normalize = false;

                Rpp32f *test_inputf32 = (Rpp32f *)calloc(srcDims[0].width * srcDims[0].height, sizeof(Rpp32f));
                Rpp32f *test_outputf32 = (Rpp32f *)calloc(numFilter * srcDims[0].width, sizeof(Rpp32f));
                // read_spectrogram(test_inputf32, srcDims, noOfAudioFiles, "spectrogram", 0, audioNames);

                start_omp = omp_get_wtime();
                start = clock();
                if (ip_bitDepth == 2)
                {
                    rppt_mel_filter_bank_host(test_inputf32, srcDescPtr, test_outputf32, dstDescPtr, srcDims, maxFreq, minFreq, melFormula, numFilter, sampleRate, normalize);
                }
                else
                    missingFuncFlag = 1;

                free(srcDims);
                free(test_inputf32);
                free(test_outputf32);

                break;
            }
            default:
            {
                missingFuncFlag = 1;
                break;
            }
        }

        end = clock();
        end_omp = omp_get_wtime();

        if (missingFuncFlag == 1)
        {
            printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
            return -1;
        }

        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        omp_time_used = end_omp - start_omp;
        if (cpu_time_used > max_time_used)
            max_time_used = cpu_time_used;
        if (cpu_time_used < min_time_used)
            min_time_used = cpu_time_used;
        avg_time_used += cpu_time_used;
    }

    avg_time_used /= 100;

    // Display measured times
    cout << fixed << "\nmax,min,avg = " << max_time_used << "," << min_time_used << "," << avg_time_used << endl;

    rppDestroyHost(handle);

    // Free memory
    free(inputAudioSize);
    free(srcLengthTensor);
    free(channelsTensor);
    free(inputf32);
    free(outputf32);

    return 0;
}
