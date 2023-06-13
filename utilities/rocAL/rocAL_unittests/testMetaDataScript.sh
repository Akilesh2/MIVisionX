cwd=$(pwd)
if [ -d build ];then 
    sudo rm -rf ./build/*
else
    mkdir build
fi
cd build || exit
cmake ..
make -j"$(nproc)"

if [[ $ROCAL_DATA_PATH == "" ]]
then 
    echo "Need to export ROCAL_DATA_PATH"
    exit
fi

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

./rocAL_unittests 2 "$coco_detection_path" "${output_path}Gamma_${rgb_name[$rgb]}_${device_name}" $width $height 33 $device $rgb 0 $display
./rocAL_unittests 5 "$tf_detection_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_notlarger_tfDetection" $width $height 0 $device $rgb 0 $display 1 3
./rocAL_unittests 7 "$caffe_detection_path" "${output_path}Saturation_${rgb_name[$rgb]}_${device_name}" $width $height 49 $device $rgb 0 $display
./rocAL_unittests 9 "$caffe2_detection_path" "${output_path}FishEye_${rgb_name[$rgb]}_${device_name}" $width $height 10 $device $rgb 0 $display

# ./rocAL_unittests 11 "$mxnet_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_mxnet" $width $height 25 $device $rgb 0 $display

./rocAL_unittests 2 /media/trial/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/ sample 640 480 32 0 1 0 1
echo Checking coco  labels
python "$cwd"/compare.py "$cwd"/meta_data_golden_outputs/labels_golden.txt "$cwd"/rocal_outputs/coco_labels.txt

echo Checking coco bbox
python "$cwd"/compare.py "$cwd"/meta_data_golden_outputs/coco_bbox_golden.txt "$cwd"/rocal_outputs/coco_bbox.txt

echo Checking caffe  labels
python "$cwd"/compare.py "$cwd"/meta_data_golden_outputs/labels_golden.txt "$cwd"/rocal_outputs/caffe_labels.txt

echo Checking caffe bbox
python "$cwd"/compare.py "$cwd"/meta_data_golden_outputs/caffe_bbox_golden.txt "$cwd"/rocal_outputs/caffe_bbox.txt

# caffe_bbox_golden
# echo Checking caffe2 labels
# python "$cwd"/compare.py "$cwd"/meta_data_golden_outputs/labels_golden.txt "$cwd"/rocal_outputs/caffe2_labels.txt

# echo Checking caffe2 bbox
# python "$cwd"/compare.py "$cwd"/meta_data_golden_outputs/caffe2_bbox_golden.txt "$cwd"/rocal_outputs/caffe2_bbox.txt

echo Checking tf labels
python "$cwd"/compare.py "$cwd"/meta_data_golden_outputs/labels_golden.txt "$cwd"/rocal_outputs/tf_labels.txt

echo Checking caffe bbox
python "$cwd"/compare.py "$cwd"/meta_data_golden_outputs/tf_bbox_golden.txt "$cwd"/rocal_outputs/tf_bbox.txt
