[example_results]: ./images/example_results.png
![example_results]
<p align = 'center'>
    <small>Example results on MS-COCO and NUS-WIDE "__with__" and "__without__" knowledge distillation using our proposed framework. The texts on the right are the top-3 predictions, where correct ones are shown in blue and incorrect in red. The green bounding boxes in images are the top-10 proposals detected by the weakly-supervised detection model.</small>
</p>

# Abstract   

Multi-label image classification is a fundamental but challenging task towards general visual understanding. Existing methods found the region-level cues (e.g., features from RoIs) can facilitate multi-label classification. Nevertheless, such methods usually require laborious object-level annotations (i.e., object labels and bounding boxes) for effective learning of the object-level visual features. In this paper, we propose a novel and efficient deep framework to boost multi-label classification by distilling knowledge from weakly-supervised detection task ___without bounding box annotations___. Specifically, given the image-level annotations, (1) we first develop a weakly-supervised detection (WSD) model, and then (2) construct an end-to-end multi-label image classification framework augmented by a knowledge distillation module that guides the classification model by the WSD model according to the class-level predictions for the whole image and the object-level visual features for object RoIs. The WSD model is the ___teacher___ model and the classification model is the ___student___ model. After this ___cross-task knowledge distillation___, the performance of the classification model is significantly improved and the efficiency is maintained since the WSD model can be safely discarded in the test phase. Extensive experiments on two large-scale datasets (MS-COCO and NUS-WIDE) show that our framework achieves superior performances over the state-of-the-art methods on both performance and efficiency.

# Motivation

[motivation]: ./images/motivation.jpg
![motivation]
<p align = 'center'>
<small>The illustration of multi-label image classification (MLIC) and weakly-supervised detection (WSD). We show top-3 predictions, in which correct predictions are shown in blueand incorrect pre-dictions inred. The MLIC model might not predict well due to poor localization for semantic instances. Although the detection results of WSD may not preserve object boundaries well, they tend to lo-cate the semantic regions which are informative for classifying the target object, such that the predictions can still be improved.</small>
</p>

## Results

[image_results]: ./figures/image_results.png
![image_results]
<p align = 'center'><small>Exemplar stylized results by the proposed Avatar-Net.</small></p>

We demonstrate the state-of-the-art effectiveness and efficiency of the proposed method in generating high-quality stylized images, with a series of successful applications including multiple style integration, video stylization and etc.

#### Comparison with Prior Arts

<p align='center'><img src="figures/closed_ups.png" width="600"></p>

- The result by Avatar-Net receives concrete multi-scale style patterns (e.g. color distribution, brush strokes and circular patterns in the style image).
- WCT distorts the brush strokes and circular patterns. AdaIN cannot even keep the color distribution, while style-swap fails in this example.

#### Execution Efficiency

<div style="padding-top: 20px; padding-bottom: 20px;">
<table>
<tbody align="center">
<tr>
<td>Method</td>
<td>Gatys et. al.</td>
<td>AdaIN</td>
<td>WCT</td>
<td>Style-Swap</td>
<td>Avatar-Net</td>
</tr>
<tr>
<td>256x256 (sec)</td>
<td>12.18</td>
<td>0.053</td>
<td>0.62</td>
<td>0.064</td>
<td>0.071</td>
</tr>
<tr>
<td>512x512 (sec)</td>
<td>43.25</td>
<td>0.11</td>
<td>0.93</td>
<td>0.23</td>
<td>0.28</td>
</tr>
</tbody>
</table>
</div>

- Avatar-Net has a comparable executive time as AdaIN and GPU-accelerated Style-Swap, and is much faster than WCT and the optimization-based style transfer by Gatys _et. al._.
- The reference methods and the proposed Avatar-Net are implemented on a same TensorFlow platform with a same VGG network as the backbone.

### Applications
#### Multi-style Interpolation
[style_interpolation]: ./figures/style_interpolation.png
![style_interpolation]

#### Content and Style Trade-off
[trade_off]: ./figures/trade_off.png
![trade_off]

#### Video Stylization ([the Youtube link](https://youtu.be/amaeqbw6TeA))

<div style="position:relative;padding-bottom:56.25%;padding-top:25px;height:0;">
<iframe style="position:absolute;width:100%;height:100%;" align="center" src="https://www.youtube.com/embed/amaeqbw6TeA" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
</div>

## Code

Please refer to the [GitHub repository](https://github.com/LucasSheng/avatar-net) for more details. 

## Publication

Lu Sheng, Ziyi Lin, Jing Shao and Xiaogang Wang, "Avatar-Net: Multi-scale Zero-shot Style Transfer by Feature Decoration", in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.  [[Arxiv](https://arxiv.org/abs/1805.03857)]

```
@inproceedings{sheng2018avatar,
    Title = {Avatar-Net: Multi-scale Zero-shot Style Transfer by Feature Decoration},
    author = {Sheng, Lu and Lin, Ziyi and Shao, Jing and Wang, Xiaogang},
    Booktitle = {Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on},
    pages={1--9},
    year={2018}
}
```
