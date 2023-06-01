from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import itertools

import torch
import numpy as np
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
# import rali_pybind.tensor
import sys
import cv2
import os
from math import sqrt
import ctypes

class ROCALCOCOIterator(object):
    """
    COCO ROCAL iterator for pyTorch.
    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, device="cpu", display=False, num_anchors=8732, pixelwise_mask=False, select_mask_id=[0], polygon_mask=False):

        try:
            assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)
        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.device = device
        self.device_id = self.loader._device_id
        self.bs = self.loader._batch_size
        self.num_anchors = num_anchors
        self.display = True
        self.pixelwise_mask = pixelwise_mask
        self.polygon_mask = polygon_mask
        self.select_mask_id = select_mask_id

        #Image id of a batch of images
        self.image_id = np.zeros(self.bs, dtype="int32")
        # Count of labels/ bboxes in a batch
        self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        # Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2), dtype="int32")
        #print("INIT exit")

    def next(self):
        return self.__next__()

    def __next__(self):
        if(self.loader.isEmpty()):
            raise StopIteration
        if self.loader.rocalRun() != 0:
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.rocalGetOutputTensors()

        #From init

        self.lis = []  # Empty list for bboxes
        self.lis_lab = []  # Empty list of labels
        self.w = self.output_tensor_list[0].batch_width()
        self.h = self.output_tensor_list[0].batch_height()
        self.bs = self.output_tensor_list[0].batch_size()
        self.color_format = self.output_tensor_list[0].color_format()

        torch_gpu_device = torch.device('cuda', self.device_id)

        #NHWC default for now
        self.out = torch.empty((self.bs, self.h, self.w, self.color_format,), dtype=torch.float32, device=torch_gpu_device)
        self.output_tensor_list[0].copy_data(ctypes.c_void_p(self.out.data_ptr()))

        # 1D labels & bboxes array
        bbox_cpu = self.loader.rocalGetBoundingBoxCords()
        # bbox_arr = torch.as_tensor(bbox_cpu, dtype=torch.float32, device=torch_gpu_device)
        label_cpu = self.loader.rocalGetBoundingBoxLabel()
        # label_arr = torch.as_tensor(label_cpu, dtype=torch.int32, device=torch_gpu_device)
        pixelwiselabel_cpu = self.loader.rocalGetPixelwiseLabels()
        # if pixelwiselabels = True
        if self.pixelwise_mask == True:
            random_mask_pixel_cpu = self.loader.rocalRandomMaskPixel()
        elif self.polygon_mask == True:
            num_objects_per_batch = self.loader.rocalGetBoundingBoxCount()
            print("num_objects_per_batch(boundingbox count ) ",num_objects_per_batch)
            mask_count_array = np.zeros(num_objects_per_batch, dtype="int32")
            print("mask_count_array ",mask_count_array)
            total_mask_count_per_batch = self.loader.rocalGetMaskCount(mask_count_array)
            print("After mask_count_array ",mask_count_array)

            print("total_mask_count_per_batch(mask  count) ",total_mask_count_per_batch)

            polygon_size_array = np.zeros(total_mask_count_per_batch, dtype="int32")
            print("after polygon_size_array ",polygon_size_array)
            mask_coordinates_list = self.loader.rocalGetMaskCoordinates(polygon_size_array, mask_count_array)
            print("polygon_size_array ",polygon_size_array)

            print("mask_coordinates_list ",len(mask_coordinates_list[0]),"   ",mask_coordinates_list)

            self.select_mask_id = [0] #Given by the user
            select_mask_array = self.loader.rocalSelectMask(self.select_mask_id)

        # Image id of a batch of images
        self.loader.GetImageId(self.image_id)
        # Image sizes of a batch
        self.loader.GetImgSizes(self.img_size)
        image_id_tensor = torch.tensor(self.image_id, device=torch_gpu_device)
        image_size_tensor = torch.tensor(self.img_size, device=torch_gpu_device).view(-1, self.bs, 2)

        if self.pixelwise_mask == True:
            return self.out, bbox_cpu, label_cpu, pixelwiselabel_cpu, random_mask_pixel_cpu
        elif self.polygon_mask == True:
            return self.out, bbox_cpu, label_cpu, pixelwiselabel_cpu, select_mask_array,num_objects_per_batch,total_mask_count_per_batch
        else:
            return self.out, bbox_cpu, label_cpu, pixelwiselabel_cpu

    def reset(self):
        self.loader.rocalResetLoaders()

    def __iter__(self):
        return self

def draw_patches(img, idx, bboxes, device):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    if device == "cpu":
        image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    htot, wtot, _ = img.shape

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for (xc, yc, w, h) in bboxes:
        l = xc - 0.5*(w)
        t = yc - 0.5*(h)
        r = xc + 0.5*(w)
        b = yc + 0.5*(h)
        loc_ = [l, t, r, b]
        color = (255, 0, 0)
        thickness = 2
        image = cv2.UMat(image).get()
        image = cv2.rectangle(image, (int(loc_[0]*wtot), int(loc_[1] * htot)), (int(
            (loc_[2] * wtot)), int((loc_[3] * htot))), color, thickness)
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/COCO_READER/" + str(idx)+"_"+"train"+".png", image)

def main():
    if  len(sys.argv) < 3:
        print ('Please pass image_path annotation_path cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/COCO_READER/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    image_path = sys.argv[1]
    annotation_path = sys.argv[2]
    if(sys.argv[3] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=300

    local_rank = 0
    world_size = 1

    rali_device = 'gpu'
    decoder_device = 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

    print("*********************************************************************")

    coco_train_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)
    pixelwise_mask = False # pixelwise_mask & polygon_mask are mutually exclusive
    polygon_mask = True # pixelwise_mask & polygon_mask are mutually exclusive
    with coco_train_pipeline:
        if pixelwise_mask == True:
            jpegs, bboxes, labels, pixelwisemaks = fn.readers.coco(file_root=image_path, annotations_file=annotation_path, pixelwise_mask = True, random_shuffle=False, shard_id=local_rank, num_shards=world_size,seed=random_seed, is_box_encoder=False, is_foreground=True, polygon_mask=False)
        if polygon_mask == True:
            jpegs, bboxes, labels, polygonmask = fn.readers.coco(file_root=image_path, annotations_file=annotation_path, pixelwise_mask = False, random_shuffle=False, shard_id=local_rank, num_shards=world_size,seed=random_seed, is_box_encoder=False, is_foreground=False, polygon_mask=True)
        else:
            jpegs, bboxes, labels = fn.readers.coco(file_root=image_path, annotations_file=annotation_path, pixelwise_mask = False, random_shuffle=False, shard_id=local_rank, num_shards=world_size,seed=random_seed, is_box_encoder=False, is_foreground=False, polygon_mask=False)

        print("*********************** SHARD ID ************************",local_rank)
        print("*********************** NUM SHARDS **********************",world_size)
        images_decoded = fn.decoders.image(jpegs, file_root=image_path, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False, annotations_file=annotation_path)
        res_images = fn.resize(images_decoded, device=rali_device, resize_width=crop, resize_height=crop, rocal_tensor_layout = types.NHWC, rocal_tensor_output_type = types.UINT8)
        images = fn.crop_mirror_normalize(res_images, device="cpu",
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mirror=0,
                                            rocal_tensor_layout = types.NHWC,
                                            rocal_tensor_output_type = types.FLOAT,
                                            mean=[0.485*255,0.456*255 ,0.406*255 ],
                                            std=[0.229*255 ,0.224*255 ,0.225*255 ])

        coco_train_pipeline.set_outputs(images)
    coco_train_pipeline.build()
    COCOIteratorPipeline = ROCALCOCOIterator(coco_train_pipeline, pixelwise_mask=pixelwise_mask, select_mask_id=[0], polygon_mask=polygon_mask)
    cnt = 0
    for epoch in range(1):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , it in enumerate(COCOIteratorPipeline):
            # print("************************************** i *************************************",i)
            # bbox= it[1][i]
            # print("bbox shape ", len(bbox))
            # print("checking  ",it[1])
            print("bounding box count ", it[5])
            print("Mask  count ", it[6])
            print("######################################")
            # for img in it[2]:
            #     print(img.shape)
            #     cnt = cnt + 1
                #draw_patches(img, cnt, "cpu")
        COCOIteratorPipeline.reset()
    print("*********************************************************************")

if __name__ == '__main__':
    main()
