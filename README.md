Multi-Label Image Classification via Knowledge Distillation from Weakly-Supervised Detection    
===
by Yongcheng Liu, Lu Sheng, Jing Shao*, Junjie Yan, Shiming Xiang and Chunhong Pan.  

![](https://github.com/Yochengliu/MLIC-KD-WSD/raw/master/images/fig3_v2.jpg)   

## Weakly Supervised Detection (WSD)   
We use WSDDN ![](http://latex.codecogs.com/gif.latex?^{[1]}) as the detection model, *i.e.*, the teacher model. Because the released code of WSDDN is Matlab (using MatConvNet), we first reproduce this paper using Caffe.

## Multi-Label Image Classification (MLIC)   
The MLIC model in our framework, *i.e.*, the student model, is very compact for efficiency. It is constituted by a popular CNN model (VGG16, as the backbone model) following a fully connected layer (as the classifier).

## Caffe Framework 
The Caffe Framework for our implementation is based on the caffe version in [PSPNet](https://github.com/hszhao/PSPNet).      
#### Installation
Please follow the instructions of [Caffe](https://github.com/BVLC/caffe) and [PSPNet](https://github.com/hszhao/PSPNet).  
The code has been tested successfully on Ubuntu 14.04 with CUDA 8.0.    

## References
[1]. Hakan Bilen, Andrea Vedaldi, 2016. Weakly Supervised Deep Detection Networks. In: IEEE Computer Vision and Pattern Recognition.   

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
