
# Sketchify
U-Net: Binary to Sketch

An attempt at expanding [deepcolor](https://github.com/kvfrans/deepcolor) to refining drawings; maps a binary image (pixel values 0, 255) to a greyscale tone image (a "sketch" with tones between 0 and 255). The network is trained with a dataset of ~5800 images collected from [/r/awwnime](https://reddit.com/r/awwnime) for 80 epoches on a TITAN X (total time ~ 5 hours).

As a proof-of-concept, results were so-so; I think they are better than any traditional techniques, but could still be improved a lot. 

<p align="center">
<img src="https://i.imgur.com/XQ2IEtb.png"><img src="https://i.imgur.com/ZeqMX1z.png"><img src="https://i.imgur.com/hn6tUjs.png">
</p>

<p align="center">
  Top: Original binarized image. Middle: Sketchify applied. Bottom: Originals (ground truth)
</p>

Implementation branched from [here](https://github.com/kvfrans/deepcolor/blob/master/main.py). Updated to Python 3.5 & tensorflow-gpu 1.2.1 syntax.

<h5>Dependencies:</h5>

Deep Learning/Image processing: Tensorflow, OpenCV (cv2), PIL, numpy, matplotlib <br/>
Data collection: praw, requests, BeautifulSoup

<h5>References:</h5>

[1] [Deepcolor: Outline Colorization through Tandem Adversarial Networks.](https://arxiv.org/pdf/1704.08834.pdf) <br/>
[2] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)