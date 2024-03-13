# FDN-MVS
The official implementation of "Feature Distribution Normalization Network for Multi-View Stereo".

> Feature Distribution Normalization Network for Multi-View Stereo <br>
> [Ziyang Chen](https://orcid.org/0000-0002-9361-0240), [Wenting Li](https://www.gzcc.edu.cn/jsjyxxgcxy/contents/3205/3569.html), [Yang Zhao](https://orcid.org/0009-0002-1031-5260), [Junling He](https://orcid.org/0009-0000-7588-3088), [Yujie Lu](https://orcid.org/0009-0008-9786-5946), [Zhongwei Cui](https://tongzhan.gznc.edu.cn/info/1015/3622.htm), [Yongjun Zhang](https://orcid.org/0000-0002-7534-1219)✱ <br>
> Visual Computer 2024 <br>
> Correspondence: ziyangchen2000@gmail.com; zyj6667@126.com✱

```bibtex
@article{fdn,
  title={Feature Distribution Normalization Network for Multi-View Stereo},
  author={Chen, Ziyang and Li, Wenting and Zhao, Yang and He, Junling and Lu, Yujie and Cui, Zhongwei and Zhang, Yongjun},
  journal={The Visual Computer},
  year={2024},
  publisher={Springer}
}
```

<div align="center">
  <img width="1600", src="./dtu.jpg">
</div>

## Model Zoo
|  Dataset   | Weight  |  Condition  |
|  :----:  | :----:  |:----:  |
|  DTU  | <a href="https://github.com/ZYangChen/FDN-MVS/releases/download/checkpoints/DTU.ckpt">44MB</a>  | 1 * NIVIDA Telsa A6000  |
|  Tanks & Temples  | <a href="https://github.com/ZYangChen/FDN-MVS/releases/download/checkpoints/TanksTemples.ckpt">44MB</a>  | 1 * NIVIDA Telsa A6000  |

## Environment Preparation
Python 3.8

PyTorch 2.0.0

CUDA 11.8

## Data Preparation

* Download pre-processed datasets (provided by PatchmatchNet): [DTU's evaluation set](https://drive.google.com/file/d/1jN8yEQX0a-S22XwUjISM8xSJD39pFLL_/view?usp=sharing), [Tanks & Temples](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view?usp=sharing)
```
root_directory
├──scan1 (scene_name1)
├──scan2 (scene_name2) 
      ├── images                 
      │   ├── 00000000.jpg       
      │   ├── 00000001.jpg       
      │   └── ...                
      ├── cams_1                   
      │   ├── 00000000_cam.txt   
      │   ├── 00000001_cam.txt   
      │   └── ...                
      └── pair.txt  
```

Camera file ``cam.txt`` stores the camera parameters, which includes extrinsic, intrinsic, minimum depth and maximum depth:
```
extrinsic
E00 E01 E02 E03
E10 E11 E12 E13
E20 E21 E22 E23
E30 E31 E32 E33

intrinsic
K00 K01 K02
K10 K11 K12
K20 K21 K22

DEPTH_MIN DEPTH_MAX 
```
``pair.txt `` stores the view selection result. For each reference image, 10 best source views are stored in the file:
```
TOTAL_IMAGE_NUM
IMAGE_ID0                       # index of reference image 0 
10 ID0 SCORE0 ID1 SCORE1 ...    # 10 best source images for reference image 0 
IMAGE_ID1                       # index of reference image 1
10 ID0 SCORE0 ID1 SCORE1 ...    # 10 best source images for reference image 1 
...
``` 

* In ``test.sh``, set `DTU_TESTING`, or `TANK_TESTING` as the root directory of corresponding dataset, set `--OUT_DIR` as the directory to store the reconstructed point clouds, uncomment the evaluation command for corresponding dataset (default is to evaluate on DTU's evaluation set)
* `CKPT_FILE` is the checkpoint file (our pretrained model is `checkpoints/DTU.ckpt` and `checkpoints/TANK_train_on_dtu.ckpt`), change it if you want to use your own model. 
* Test on GPU by running `sh test.sh`. The code includes depth map estimation and depth fusion. The outputs are the point clouds in `ply` format. 
* For quantitative evaluation on DTU dataset, download [SampleSet](http://roboimagedata.compute.dtu.dk/?page_id=36) and [Points](http://roboimagedata.compute.dtu.dk/?page_id=36). Unzip them and place `Points` folder in `SampleSet/MVS Data/`. The structure looks like:
```
SampleSet
├──MVS Data
      └──Points
```

The performance on Tanks & Temples datasets will be better if the model is fine-tuned on BlendedMVS Datasets

* Download the BlendedMVS [dataset](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDgxb8MDGgoV74S?e=hJKlvV).

* For detailed quantitative results on Tanks & Temples, please check the leaderboards [Tanks & Temples](https://www.tanksandtemples.org/leaderboard/AdvancedF/)

* In ``train.sh``, set `MVS_TRAINING` or `BLEND_TRAINING` as the root directory of dataset; set `--logdir` as the directory to store the checkpoints. 
* Train the model by running `sh train.sh`.

DTU Training dataset:  
Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
 and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) 
 (both from [Original MVSNet](https://github.com/YoYo000/MVSNet)), and upzip it as the $MVS_TRANING  folder.

