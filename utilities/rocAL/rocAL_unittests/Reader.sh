echo $ROCAL_DATA_PATH
cd build
# ls
path=($ROCAL_DATA_PATH/coco/coco_10_img/train_10images_2017/ 
    $ROCAL_DATA_PATH/coco/coco_10_img/train_10images_2017/ 
    $ROCAL_DATA_PATH/coco/coco_10_img/train_10images_2017/  
    $ROCAL_DATA_PATH/coco/coco_10_img/train_10images_2017/
    $ROCAL_DATA_PATH/tf/classification/
    $ROCAL_DATA_PATH/tf/detection/
    $ROCAL_DATA_PATH/caffe/classification/ilsvrc12_train_lmdb/
    $ROCAL_DATA_PATH/caffe/detection/lmdb_record/
    $ROCAL_DATA_PATH/caffe2/classfication/imagenet_val5_encode/
    $ROCAL_DATA_PATH/caffe2/detection/lmdb_records/
    $ROCAL_DATA_PATH/coco/coco_10_img/train_10images_2017/)

#coco_reader =$ROCAL_DATA_PATH/coco/coco_10_img/train_10images_2017/
# coco_partical_reader = $ROCAL_DATA_PATH/coco/coco_10_img/train_10images_2017/
# tf_classification = $ROCAL_DATA_PATH/tf/classification/
# tf_detection =  $ROCAL_DATA_PATH/tf/detection/
# caffe_classification = $ROCAL_DATA_PATH/caffe/classification/ilsvrc12_train_lmdb/
#caffe_detection = $ROCAL_DATA_PATH/caffe/detection/lmdb_record/
#caffe2_classification =  $ROCAL_DATA_PATH/caffe2/classfication/imagenet_val5_encode/  
# caffe2_detection = $ROCAL_DATA_PATH/caffe2/detection/lmdb_records/
# coco+reader_keypoints = $ROCAL_DATA_PATH/coco/coco_10_img/train_10images_2017/
 

echo ${path[0]}
output_file="sample"
resize_width=300
resize_height=300
test_case=0
rgb=1
one_hot_labels=0
display_all=1



# exit
# ./rocAL_unittests 4 1 /media/MIVisionX-data/rocal_data/caffe/classification/ilsvrc12_train_lmdb/ sample 300 300 0 0 1 0 0
for reader_type in 0 1 2 3 4 5 6 7 8 9 
do
    echo reader_typeee $reader_type
    for pipeline in 1 2
    do 
        echo pipelineee $pipeline
        for  device_type in 0 1 
        do 
            echo device_typeee $device_type
            echo ./rocAL_unittests $reader_type $pipeline ${path[$reader_type]} $output_file $resize_height $resize_width $test_case $device_type $rgb $one_hot_labels $display_all
            ./rocAL_unittests $reader_type $pipeline ${path[$reader_type]} $output_file $resize_height $resize_width $test_case $device_type $rgb $one_hot_labels $display_all
        done
    done
done    