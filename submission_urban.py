# -*- coding: utf-8 -*-
''' 
The order of LF image files may be different with this file.
(Top to Bottom, Left to Right, and so on..)

If you use different LF images, 

you should change our 'func_makeinput.py' file.

# Light field images: input_Cam000-080.png
# All viewpoints = 9x9(81)

# -- LF viewpoint ordering --
# 00 01 02 03 04 05 06 07 08
# 09 10 11 12 13 14 15 16 17
# 18 19 20 21 22 23 24 25 26
# 27 28 29 30 31 32 33 34 35
# 36 37 38 39 40 41 42 43 44
# 45 46 47 48 49 50 51 52 53
# 54 55 56 57 58 59 60 61 62
# 63 64 65 66 67 68 69 70 71
# 72 73 74 75 76 77 78 79 80

'''

import numpy as np
import os
import time
from LF_func.func_pfm import write_pfm, read_pfm
from LF_func.func_makeinput import make_epiinput
from LF_func.func_makeinput import make_input
from LF_func.contextnet import define_contextnet

# import matplotlib.pyplot as plt
import cv2
import imageio


def save_disparity_jet(disparity, filename):
    max_disp = np.nanmax(disparity[disparity != np.inf])
    min_disp = np.nanmin(disparity[disparity != np.inf])
    disparity = (disparity - min_disp) / (max_disp - min_disp)
    disparity = (disparity * 255.0).astype(np.uint8)
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    cv2.imwrite(filename, disparity)


if __name__ == '__main__':

    dir_output = 'submission_contextnet_urban'

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # GPU setting ( rtx 3090 - gpu0 )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    '''
    /// Setting 1. LF Images Directory

    LFdir = 'synthetic': Test synthetic LF images (from 4D Light Field Benchmark)
                                   "A Dataset and Evaluation Methodology for 
                                   Depth Estimation on 4D Light Fields".
                                   http://hci-lightfield.iwr.uni-heidelberg.de/

    '''
    dir_LFimages = [
        'UrbanLF-Syn/test/' + i for i in sorted(os.listdir('UrbanLF-Syn/test'),
                                                key=lambda x: int(x[5:]))
    ]

    image_h = 480
    image_w = 640

    # number of views ( 0~8 for 9x9 )
    AngualrViews = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    path_weight = 'ptrtrain_contextnet.hdf5'

    img_scale = 1  # 1 for small_baseline(default) <3.5px,
    # 0.5 for large_baseline images   <  7px

    img_scale_inv = int(1 / img_scale)
    ''' Define Model ( set parameters )'''

    model_learning_rate = 0.001
    crop = True
    if crop:
        model_512 = define_contextnet(320, 320, AngualrViews,
                                      model_learning_rate)
    else:
        model_512 = define_contextnet(image_w, image_h, AngualrViews,
                                      model_learning_rate)
    ''' Model Initialization '''

    model_512.load_weights(path_weight)
    print('load pretrain model!!!')
    dum_sz = model_512.input_shape[0]
    dum = np.zeros((1, dum_sz[1], dum_sz[2], dum_sz[3]), dtype=np.float32)
    tmp_list = []
    for i in range(81):
        tmp_list.append(dum)
    dummy = model_512.predict(tmp_list, batch_size=1)
    """  Depth Estimation  """
    for image_path in dir_LFimages:
        print(image_path)
        val_list = make_input(image_path, image_h, image_w, AngualrViews)

        start = time.time()

        if crop:
            crop_size = 320
            stride = 160
            val_output_tmp = np.zeros((1, image_h, image_w), dtype=np.float32)
            val_mask_weight = np.zeros((1, image_h, image_w), dtype=np.float32)
            for i in range(6):
                (row, colume) = divmod(i, 3)
                start_h = row * stride
                start_w = colume * stride
                crop_valdata = [
                    val[:, start_h:start_h + crop_size,
                        start_w:start_w + crop_size, :] for val in val_list
                ]
                val_crop_output_tmp = model_512.predict(crop_valdata,
                                                        batch_size=1)
                val_output_tmp[:, start_h:start_h + crop_size,
                               start_w:start_w +
                               crop_size] += val_crop_output_tmp[:, :, :]
                val_mask_weight[:, start_h:start_h + crop_size,
                                start_w:start_w + crop_size] += 1
            val_output_tmp = val_output_tmp / val_mask_weight
        else:
            val_output_tmp = model_512.predict(val_list, batch_size=1)

        runtime = time.time() - start
        # plt.imshow(val_output_tmp[0, :, :])

        save_disparity_jet(
            val_output_tmp[0, :, :],
            dir_output + '/%s.jpg' % (image_path.split('/')[-1]))
        if not os.path.exists(dir_output + '/' + image_path.split('/')[-1]):
            os.makedirs(dir_output + '/' + image_path.split('/')[-1])
        np.save(
            os.path.join(dir_output,
                         image_path.split('/')[-1], 'disp.npy'),
            val_output_tmp[0, :, :])
        # write_pfm(
        #     val_output_tmp[0, :, :],
        #     dir_output + '/disp_maps/%s.pfm' % (image_path.split('/')[-1]))
        # print('pfm file saved in %s/%s.pfm' %
        #       (dir_output, image_path.split('/')[-1]))
