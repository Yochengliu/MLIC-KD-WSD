Multi-Label Image Classification via Knowledge Distillation from Weakly-Supervised Detection    
===
This repository contains the code (in [Caffe](https://github.com/BVLC/caffe)) for the paper:

[__Multi-Label Image Classification via Knowledge Distillation from Weakly-Supervised Detection__](https://arxiv.org/abs/1809.05884)
<br>
[Yongcheng Liu](https://www.researchgate.net/profile/Yongcheng_Liu), [Lu Sheng](http://www.ee.cuhk.edu.hk/~lsheng/), [Jing Shao*](https://amandajshao.github.io/), [Junjie Yan](http://www.cbsr.ia.ac.cn/users/jjyan/main.htm), [Shiming Xiang](https://scholar.google.com/citations?user=0ggsACEAAAAJ&hl=zh-CN) and [Chunhong Pan](http://people.ucas.ac.cn/~0005314)
<br>
[_ACM Multimedia 2018_](http://www.acmmm.org/2018/)

__Project Page__: [https://yochengliu.github.io/MLIC-KD-WSD/](https://yochengliu.github.io/MLIC-KD-WSD/)

## Weakly Supervised Detection (WSD)
 
- We use [WSDDN](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bilen_Weakly_Supervised_Deep_CVPR_2016_paper.pdf) ![](http://latex.codecogs.com/gif.latex?^{[1]}) as the detection model, *i.e.*, the teacher model. 

- Because the [released code](https://github.com/hbilen/WSDDN) of WSDDN is implemented using Matlab (based on [MatConvNet](http://www.vlfeat.org/matconvnet/)), we first reproduce this paper using Caffe.

[1]. Hakan Bilen, Andrea Vedaldi, "Weakly Supervised Deep Detection Networks". In: IEEE Computer Vision and Pattern Recognition, 2016.

#### Reproduction results

**_detection_**  

[wsddn_det]: ./docs/images/wsddn_det.jpg
![wsddn_det]

- __Paper__ 

        training: 5 scales + mirror          testing: fusion of 5 scales + mirror

- __Our__   

        training: 5 scales + mirror          testing: single-forward test

**_classification_**

[wsddn_cls]: ./docs/images/wsddn_cls.jpg
![wsddn_cls]

#### Datalist Preparation

    image_path one_hot_label_vector(e.g., 0 1 1 ...) proposal_info(e.g., x_min y_min x_max y_max score x_min y_min x_max y_max score ...)

#### Training & Test

        ./wsddn/wsddn_train(deploy).prototxt
    
- [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) is used as the backbone model.

- For training, we did not use spatial regularizer. More details can be referred in the paper.

- For testing, you can use Pycaffe or Matcaffe.

## Multi-Label Image Classification (MLIC)
   
- The MLIC model in our framework, *i.e.*, the student model, is very compact for efficiency.

- It is constituted by a popular CNN model ([VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/), as the backbone model) following a fully connected layer (as the classifier).

- The backbone model of the student could be different from the teacher's.

## Cross-Task Knowledge Distillation

#### Stage 1: Feature-Level Knowledge Transfer

        ./kd/train_stage1.prototxt

#### Stage 2: Prediction-Level Knowledge Transfer

        ./kd/train_stage2.prototxt

Datalist preparation is the same as mentioned in WSD. More details can be referred in our paper.

## Implementation

Please refer to caffe-MLIC for details.

## Citation

If our paper [[ACM DL](https://dl.acm.org/citation.cfm?id=3240567)] [[arXiv](https://arxiv.org/abs/1809.05884)] is helpful for your research, please consider citing:   

        @inproceedings{liu2018mlickdwsd,   
          author = {Yongcheng Liu and    
                    Lu Sheng and    
                    Jing Shao and   
                    Junjie Yan and   
                    Shiming Xiang and   
                    Chunhong Pan},   
          title = {Multi-Label Image Classification via Knowledge Distillation from Weakly-Supervised Detection},   
          booktitle = {ACM International Conference on Multimedia},    
          pages = {700--708},  
          year = {2018}   
        }   

## Contact

If you have some ideas or questions about our research to share with us, please contact <yongcheng.liu@nlpr.ia.ac.cn>
