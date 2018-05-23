# Top-k Recommendation
## **Introduction**</br>

A python code collection for top-k recommendation refactored by tensorflow</br>
The old repository is [here](old).</br>

The collection will consist of following methods:
- <a href="https://arxiv.org/abs/1205.2618">Bayesian Personalized Ranking (BPR)</a></br>
  - BPR is the very first version of the BPR based methods. </br>
  - It is only applicable in in-matrix recommendation scenario. </br>
- <a href="https://arxiv.org/abs/1510.01784">Visual Bayesion Personalized Ranking (VBPR)</a></br>
  - VBPR is the extension of BPR to combine visual contents in the rating prediction. </br>
  - It can recommend videos in both in-matrix and out-of-matrix recommendation scenarios. </br>
- <a href="https://papers.nips.cc/paper/5004-deep-content-based-music-recommendation">DeepMusic (DPM)</a></br>
  - DPM uses multiple layer perceprion (MLP) to learn the content latent vectors from MFCC. </br>
  - It recommends videos in both in-matrix and out-of-matrix recommendation scenarios. </br>
- <a href="http://www.cs.columbia.edu/~blei/papers/WangBlei2011.pdf">Collaborative Topic Regression (CTR)</a></br>
  - CTR uses LDA to learn the topic distribution from the textual content vectors, then performs the collaborative regression to learn the user and item latent vectors.</br>
  - CTR can perform in-matrix and out-of-matrix recommendation but only with the textual content vectors.</br>
  - The original code can be downloaded from <a href="http://www.cs.cmu.edu/~chongw/citeulike/">here</a>.</br>
- <a href="https://arxiv.org/abs/1409.2944">Collaborative Deep Learning (CDL)</a></br>
  - CDL uses stacked denoising auto-encoder (SDAE) to learn the content latent vectors, then performs the collaborative regression to learn the user and item latent vectors.</br>
  - CDL can perform in-matrix and out-of-matrix recommendation.</br>
  - The original code can be downloaded from <a href="http://www.wanghao.in/code/cdl-release.rar">here</a>.</br>
  - CDL originally supports textual contents only.</br>
  - CDL can support non-textual contents by replacing the binary visiable layer with Gaussian visiable layer.</br>
- <a href="https://arxiv.org/abs/1708.05031">Neural Collaborative Filtering (NCF)</a></br>
- <a href="">Collaborative Embedding Regression (CER)</a></br>
</br>

## **Instruction**</br>
All the code in the repository is written in Python 3.</br>
To simplify the installation of Python 3, please use Anaconda.</br>
The dependencies are [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/scipylib), [tensorflow](https://www.tensorflow.org/).</br>
After forking, you should configure several things before running the code:</br>
- Use pip to install numpy, scipy, and tensorflow;
- Download datasets </br>

For training bpr and vbpr, you can run
```
python train.py
```

## **Dataset**</br>
Due to the file size limitation, the data for training and testing are hosted by other places.</br>
The 5-fold experimental data can be downloaded from below link:</br>
<a href="https://drive.google.com/open?id=0Bz6bXb44ws2WcGtyNGltajJTcWc">Google Drive</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="http://pan.baidu.com/s/1jHPBVgy">Baidu Yunpan</a></br>
The content features can be downloaded from below link:</br>
<a href="https://drive.google.com/open?id=0Bz6bXb44ws2WUXBuVGwzNDBlQXM">Google Drive</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="http://pan.baidu.com/s/1kVuHWnP">Baidu Yunpan</a></br>
To make the code run correctly, these extenal data should be put to the project directory.</br>
The original 10380 videos can be downloaded from below link:</br>
<a href="http://pan.baidu.com/s/1jIDdAwI">Baidu Yunpan</a></br>
## **Reference**</br>
If you use above codes or data, please cite the paper below:</br>
@article{VCRS, </br>
  author    = {Xingzhong Du and Hongzhi Yin and Ling Chen and Yang Wang and Yi Yang and Xiaofang Zhou}, </br>
  title     = {Exploiting Rich Contents for Personalized Video Recommendation}, </br>
  journal   = {CoRR}, </br>
  volume    = {abs/1612.06935}, </br>
  year      = {2016} </br>
} </br>
