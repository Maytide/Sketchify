
# Sketchify
U-Net: Binary to Sketch

An attempt at expanding [deepcolor](https://github.com/kvfrans/deepcolor) to refining drawings. I collect images of anime-style drawings then apply sketch-style and binary filters to be ground truth and training data, respectively. [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_networks) implemented in tensorflow (branched from deepcolor). The network is trained with a dataset of ~5800 images collected from [/r/awwnime](https://reddit.com/r/awwnime) for 80 epoches on a TITAN X (total time ~ 5 hours).

Results were so-so; I think they are better than any traditional techniques, but could still be improved a lot. 

<p align="center">
<img src="http://i.imgur.com/w14c9Vo.jpg"><img src="http://i.imgur.com/5NIujiK.png">
</p>

<p align="center">
  Top: Original binarized image. Bottom: Sketchify applied.
</p>

Implementation branched from [here](https://github.com/kvfrans/deepcolor/blob/master/main.py). Updated to Python 3.5 & tensorflow-gpu 1.2.1 syntax.

Extra information: https://maytide.github.io/sketch/tensorflow/2017/08/28/sketchify.html

<h5>Dependencies:</h5>

Deep Learning/Image processing: Tensorflow, OpenCV (cv2), PIL, numpy, matplotlib <br/>
Data collection: praw, requests, BeautifulSoup

<h5>References:</h5>

[1] [Deepcolor: Outline Colorization through Tandem Adversarial Networks.](https://arxiv.org/pdf/1704.08834.pdf) <br/>
[2] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
