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
We use WSDDN ![](http://latex.codecogs.com/gif.latex?^{[1]}) as the detection model, *i.e.*, the teacher model. Because the released code of WSDDN is implemented using Matlab (based on MatConvNet), we first reproduce this paper using Caffe.

[1]. Hakan Bilen, Andrea Vedaldi, "Weakly Supervised Deep Detection Networks". In: IEEE Computer Vision and Pattern Recognition, 2016.

### Datalist Preparation
   
    image_path one_hot_label_vector(*e.g.*, 0 1 1 ...) proposal_info(*e.g.*, x_min y_min x_max y_max score x_min y_min x_max y_max score ...)

### Training & Test
   
    ./wsddn/wsddn_train(deploy).prototxt

For testing WSDDN, you can use Pycaffe or Matcaffe.

## Multi-Label Image Classification (MLIC)   
The MLIC model in our framework, *i.e.*, the student model, is very compact for efficiency. It is constituted by a popular CNN model (VGG16, as the backbone model) following a fully connected layer (as the classifier).



## Caffe 
The Caffe for our implementation is based on the Caffe version in [PSPNet](https://github.com/hszhao/PSPNet).      
#### Installation
Please follow the instructions of [Caffe](https://github.com/BVLC/caffe) and [PSPNet](https://github.com/hszhao/PSPNet).  
The code has been tested successfully on Ubuntu 14.04 with CUDA 8.0.    

## References


## Citation
If our paper is helpful for your research, please consider citing:   

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
