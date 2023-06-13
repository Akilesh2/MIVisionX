cwd=$(pwd)
if [ -d build ];then 
    sudo rm -rf ./build/*
else
    mkdir build
fi
if [ -d ./rocal_outputs ];then 
    echo checking if part 
    sudo rm -rf ./rocal_outputs/*
else
    echo checking else part 
    mkdir rocal_outputs
fi
cd build || exit
cmake ..
make -j"$(nproc)"

if [[ $ROCAL_DATA_PATH == "" ]]
then 
    echo "Need to export ROCAL_DATA_PATH"
    exit
fi

# if [ -d ./rocal_outputs ];then 
#     echo checking if part 
#     sudo rm -rf ./rocal_outputs/*
# else
#     echo checking else part 
#     mkdir rocal_outputs
# fi
image_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/train_10images_2017/
coco_detection_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/train_10images_2017/
tf_classification_path=${ROCAL_DATA_PATH}/rocal_data/tf/classification/
tf_detection_path=${ROCAL_DATA_PATH}/rocal_data/tf/detection/
caffe_classification_path=${ROCAL_DATA_PATH}/rocal_data/caffe/classification/
caffe_detection_path=${ROCAL_DATA_PATH}/rocal_data/caffe/detection/
caffe2_classification_path=${ROCAL_DATA_PATH}/rocal_data/caffe2/classification/
caffe2_detection_path=${ROCAL_DATA_PATH}/rocal_data/caffe2/detection/
mxnet_path=${ROCAL_DATA_PATH}/rocal_data/mxnet/
output_path=../rocal_unittest_output_folder_$(date +%Y-%m-%d_%H-%M-%S)/
golden_output_path=${ROCAL_DATA_PATH}/rocal_data/GoldenOutputs/

display=0
device=0
width=640 
height=480
device_name="host"
rgb_name=("gray" "rgb")
rgb=1
dev_start=0
dev_end=1
rgb_start=0
rgb_end=1
rgb=1
device=0

if [ "$#" -gt 0 ]; then 
    if [ "$1" -eq 0 ]; then # For only HOST backend
        dev_start=0
        dev_end=0
    elif [ "$1" -eq 1 ]; then # For only HIP backend
        dev_start=1
        dev_end=1
    elif [ "$1" -eq 2 ]; then # For both HOST and HIP backend
        dev_start=0
        dev_end=1
    fi
fi

if [ "$#" -gt 1 ]; then
    if [ "$2" -eq 0 ]; then # For only Greyscale inputs
        rgb_start=0
        rgb_end=0
    elif [ "$2" -eq 1 ]; then # For only RGB inputs
        rgb_start=1
        rgb_end=1
    elif [ "$2" -eq 2 ]; then # For both RGB and Greyscale inputs
        rgb_start=0
        rgb_end=1
    fi
fi

for ((device=dev_start;device<=dev_end;device++))
do 
    if [ $device -eq 1 ]
    then 
        device_name="hip"
        echo "Running HIP Backend..."
    else
        echo "Running HOST Backend..."
    fi
    for ((rgb=rgb_start;rgb<=rgb_end;rgb++))
    do 
        ./rocAL_unittests 2 "$coco_detection_path" "${output_path}Gamma_${rgb_name[$rgb]}_${device_name}" $width $height 33 $device $rgb 0 $display
        ./rocAL_unittests 5 "$tf_detection_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_notlarger_tfDetection" $width $height 0 $device $rgb 0 $display 1 3
        ./rocAL_unittests 7 "$caffe_detection_path" "${output_path}Saturation_${rgb_name[$rgb]}_${device_name}" $width $height 49 $device $rgb 0 $display
        # ./rocAL_unittests 9 "$caffe2_detection_path" "${output_path}FishEye_${rgb_name[$rgb]}_${device_name}" $width $height 10 $device $rgb 0 $display
    done
done

python "$cwd"/compare.py "$cwd"/meta_data_golden_outputs/ "$cwd"/rocal_outputs/