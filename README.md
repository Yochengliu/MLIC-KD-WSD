Multi-Label Image Classification via Knowledge Distillation from Weakly-Supervised Detection    
===
This repository contains the code (in [Caffe](https://github.com/BVLC/caffe)) for the paper:

[__Multi-Label Image Classification via Knowledge Distillation from Weakly-Supervised Detection__]()
<br>
[Yongcheng Liu](mailto:yongcheng.liu@nlpr.ia.ac.cn), [Lu Sheng](http://www.ee.cuhk.edu.hk/~lsheng/), [Jing Shao*](http://www.ee.cuhk.edu.hk/~jshao/), [Junjie Yan](http://www.cbsr.ia.ac.cn/users/jjyan/main.htm), [Shiming Xiang](http://www.escience.cn/people/smxiang) and [Chunhong Pan](http://people.ucas.ac.cn/~0005314)
<br>
[_ACM Multimedia 2018_](http://www.acmmm.org/2018/)

__Project Page__: [https://yochengliu.github.io/MLIC-KD-WSD/](https://yochengliu.github.io/MLIC-KD-WSD/)

## Weakly Supervised Detection (WSD)   
We use [WSDDN](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bilen_Weakly_Supervised_Deep_CVPR_2016_paper.pdf) ![](http://latex.codecogs.com/gif.latex?^{[1]}) as the detection model, *i.e.*, the teacher model. Because the released code of WSDDN is implemented using Matlab (based on MatConvNet), we first reproduce this paper using Caffe.

[1]. Hakan Bilen, Andrea Vedaldi, "Weakly Supervised Deep Detection Networks". In: IEEE Computer Vision and Pattern Recognition, 2016.

#### Datalist Preparation
    image_path one_hot_label_vector(e.g., 0 1 1 ...) proposal_info(e.g., x_min y_min x_max y_max score x_min y_min x_max y_max score ...)

#### Training & Test
        ./wsddn/wsddn_train(deploy).prototxt
    
- For training, we did not use spatial regularizer. More details can be referred in the paper.
- For testing, you can use Pycaffe or Matcaffe.

## Multi-Label Image Classification (MLIC)   
The MLIC model in our framework, *i.e.*, the student model, is very compact for efficiency. It is constituted by a popular CNN model (VGG16, as the backbone model) following a fully connected layer (as the classifier). Actually, the backbone model of the student could be different from the teacher's.

## Cross-Task Knowledge Distillation

#### Stage 1: Feature-Level Knowledge Transfer
        ./kd/train_stage1.prototxt
#### Stage 2: Prediction-Level Knowledge Transfer
        ./kd/train_stage2.prototxt

More details can be referred in our paper.

## Caffe     
#### Installation
Please follow the instruction of [Caffe](https://github.com/BVLC/caffe).  

#### Our Implementation
        ./caffe
            include
                ...
            src
                caffe
                    utils
                        interp.cpp/cu                   // bilinear interpolation
                    cross_entropy_loss_layer.cpp        // cross entropy loss for WSDDN
                    data_transformer.cpp                // data augmentation
                    human_att_data_layer.cpp            // data layer
                    roi_pooling_layer.cpp/cu            // add score
                    wsd_roigen_layer.cpp                // prepare rois for roi pooling
                    wsd_roigen_single_scale_layer.cpp   // convert rois' coordinates according to the given scale
                proto
                    caffe.proto                         // add some LayerParameters 

__Note__: You shoud add the above codes to Caffe and compile them successfully.

The code has been tested successfully on Ubuntu 14.04 with CUDA 8.0.    

## Citation
If our paper [[arXiv]()] is helpful for your research, please consider citing:   

    @inproceedings{liu2018mlickdwsd,   
      author = {Yongcheng Liu and    
                Lu Sheng and    
                Jing Shao and   
                Junjie Yan and   
                Shiming Xiang and   
                Chunhong Pan},   
      title = {Multi-Label Image Classification via Knowledge Distillation from Weakly-Supervised Detection},   
      booktitle = {ACM International Conference on Multimedia},    
      pages = {1--9},  
      year = {2018}   
    }   

## Contact
If you have some ideas or questions about our research to share with us, please contact <yongcheng.liu@nlpr.ia.ac.cn>
