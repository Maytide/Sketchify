
# Sketchify
U-Net: Binary to Sketch

An attempt at expanding [deepcolor](https://github.com/kvfrans/deepcolor) to refining drawings. I collect images of anime-style drawings then apply sketch-style and binary filters to be ground truth and training data, respectively. Implemented in tensorflow. The network is trained with a dataset of ~5800 images collected from [/r/awwnime](https://reddit.com/r/awwnime) for 80 epoches on a TITAN X (total time ~ 5 hours).

Results were so-so; I think they are better than any traditional techniques, but could certainly still be improved upon. 

<img src="http://i.imgur.com/w14c9Vo.jpg" style="max-width: 384px !important; max-height: 384px !important;"><img src="http://i.imgur.com/5NIujiK.png" style="max-width: 384px; max-height: 384px;">

Left: Original binarized image. Right: Sketchify applied.

Implementation branched from deepcolor.

<h5>Dependencies:</h5>

Image processing/Deep Learning: Tensorflow, OpenCV (cv2), PIL, numpy, matplotlib <br/>
Data collection: praw, requests, BeautifulSoup

<h5>References:</h5>

[1] [Deepcolor: Outline Colorization through Tandem Adversarial Networks.](https://arxiv.org/pdf/1704.08834.pdf) <br/>
[2] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)