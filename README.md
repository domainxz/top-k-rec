# Top-k Recommendation
## **Introduction**</br>
A method collection for top-k recommendation</br>
The collection consists of following methods:</br>
- <a href="https://arxiv.org/abs/1205.2618">Bayesian Personalized Ranking (BPR)</a></br>
<p>
BPR is the very first version of the BPR based methods. </br>
It is only applicable in in-matrix recommendation scenario.
</p>
- <a href="https://arxiv.org/abs/1510.01784">Visual Bayesion Personalized Ranking (VBPR)</a></br>
<p>
VBPR is the extension of BPR to combine visual contents in the rating prediction. </br>
It can recommend videos in both in-matrix and out-of-matrix recommendation scenarios.
</p>
- <a href="https://papers.nips.cc/paper/5004-deep-content-based-music-recommendation">DeepMusic (DPM)</a></br>
<p>
DPM uses multiple layer perceprion (MLP) to learn the content latent vectors from MFCC.</br>
It recommends videos in both in-matrix and out-of-matrix recommendation scenarios.
</p>
- <a href="">Collaborarive Embedding Regression (CER)</a><br>

Some methods have already released their source codes:
- <a href="http://www.cs.columbia.edu/~blei/papers/WangBlei2011.pdf">Collaborative Topic Regression (CTR)</a>:
<p>
CTR uses LDA to learn the topic distribution from the textual content vectors, then performs the collaborative regression to learn the user and item latent vectors.</br>
CTR can perform in-matrix and out-of-matrix recommendation but only with the textual content vectors.
The source code can be downloaded from <a href="http://www.cs.cmu.edu/~chongw/citeulike/">here</a>.
</p>
- <a href="https://arxiv.org/abs/1409.2944">Collaborative Deep Learning (CDL)</a>:
<p>
CDL uses stacked denoising auto-encoder (SDAE) to learn the content latent vectors, then performs the collaborative regression to learn the user and item latent vectors.</br>
CDL can perform in-matrix and out-of-matrix recommendation.</br>
The source code can be downloaded from <a href="http://www.wanghao.in/code/cdl-release.rar">here</a></br>.
CDL originally supports textual contents only. But we can make CDL support non-textual contents by replacing the binary visiable layer with Gaussian visiable layer.
</p>

## **Instruction**</br>
## **External Data**</br>
Due to the file size limitation, the data for training and testing are maintained on other services.</br>
The 5-fold experimental data can be downloaded from below link:</br>
<a href="https://drive.google.com/open?id=0Bz6bXb44ws2WcGtyNGltajJTcWc">Google Drive</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="http://pan.baidu.com/s/1jHPBVgy">Baidu Yunpan</a></br>
The content features can be downloaded from below link:</br>
<a href="https://drive.google.com/open?id=0Bz6bXb44ws2WUXBuVGwzNDBlQXM">Google Drive</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="http://pan.baidu.com/s/1kVuHWnP">Baidu Yunpan</a></br>

