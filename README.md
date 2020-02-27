# C3D-tensorflow

This is a repository trying to implement [C3D-caffe][5] on tensorflow,useing models directly converted from original C3D-caffe.    
Be aware that there are about 5% video-level accuracy margin on UCF101 split1  between our implement in tensorflow and  the original C3D-caffe.  

## Requirements:

1. Have installed the tensorflow >= 1.2 version
2. You must have installed the following two python libs:
a) [tensorflow][1]
b) [Pillow][2]
3. You must have downloaded your own dataset or the dataset [UCF101][3] (Action Recognition Data Set)
4. Each single avi file is decoded with 5FPS (it's depend your decision) in a single directory.
    - you can use the `./list/convert_video_to_images.sh` script to decode the ucf101 video files
    - `# ./list/convert_video_to_images.sh $dataset_dir $FPS`
    - run `./list/convert_video_to_images.sh .../UCF101 5`
5. Generate {train,test}.list files, and put them in the `list` directory. Each line corresponds to "image directory" and a class (zero-based). For example:
    - you can use the `./list/convert_images_to_list.sh` script to generate the {train,test}.list for the dataset
    - run `./list/convert_images_to_list.sh .../dataset_images 4`, this will generate `test.list` and `train.list` files by a factor 4 inside the root folder

```
database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 0
database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c02 0
database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03 0
database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c01 1
database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c02 1
database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c03 1
database/ucf101/train/Archery/v_Archery_g01_c01 2
database/ucf101/train/Archery/v_Archery_g01_c02 2
database/ucf101/train/Archery/v_Archery_g01_c03 2
database/ucf101/train/Archery/v_Archery_g01_c04 2
database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c01 3
database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c02 3
database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c03 3
database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c04 3
database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c01 4
database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c02 4
database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c03 4
database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c04 4
...
```

## Usage

1. `python train_c3d_ucf101.py` will train C3D model. The trained model will saved in `models` directory.
2. `python predict_c3d_ucf101.py` will test C3D model on a validation data set.
3. IMPORTANT NOTE: when you load the sports1m_finetuning_ucf101.model,you should use the tranpose operation like:` pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])`,or in `Random_clip_valid.py` looks like:`["transpose", [0, 1, 4, 2, 3]]`, 
but if you load `conv3d_deepnetA_sport1m_iter_1900000_TF.model` or `c3d_ucf101_finetune_whole_iter_20000_TF.model`,you don't need tranpose operation,just comment that line code.  
