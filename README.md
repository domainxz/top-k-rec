# Top-k Recommendation
## **Introduction**</br>

A method collection for top-k recommendation</br>

The collection consists of following methods:
- <a href="https://arxiv.org/abs/1205.2618">Bayesian Personalized Ranking (BPR)</a></br>
BPR is the very first version of the BPR based methods. </br>
It is only applicable in in-matrix recommendation scenario. </br>
- <a href="https://arxiv.org/abs/1510.01784">Visual Bayesion Personalized Ranking (VBPR)</a></br>
VBPR is the extension of BPR to combine visual contents in the rating prediction. </br>
It can recommend videos in both in-matrix and out-of-matrix recommendation scenarios. </br>
- <a href="https://papers.nips.cc/paper/5004-deep-content-based-music-recommendation">DeepMusic (DPM)</a></br>
DPM uses multiple layer perceprion (MLP) to learn the content latent vectors from MFCC. </br>
It recommends videos in both in-matrix and out-of-matrix recommendation scenarios. </br>
- <a href="">Collaborarive Embedding Regression (CER)</a></br>
</br>

Some methods have already released their source codes:
- <a href="http://www.cs.columbia.edu/~blei/papers/WangBlei2011.pdf">Collaborative Topic Regression (CTR)</a></br>
  - CTR uses LDA to learn the topic distribution from the textual content vectors, then performs the collaborative regression to learn the user and item latent vectors.</br>
  - CTR can perform in-matrix and out-of-matrix recommendation but only with the textual content vectors.</br>
  - The source code can be downloaded from <a href="http://www.cs.cmu.edu/~chongw/citeulike/">here</a>.</br>
- <a href="https://arxiv.org/abs/1409.2944">Collaborative Deep Learning (CDL)</a></br>
  - CDL uses stacked denoising auto-encoder (SDAE) to learn the content latent vectors, then performs the collaborative regression to learn the user and item latent vectors.</br>
  - CDL can perform in-matrix and out-of-matrix recommendation.</br>
  - The source code can be downloaded from <a href="http://www.wanghao.in/code/cdl-release.rar">here</a>.</br>
  - CDL originally supports textual contents only.</br>
  - CDL can support non-textual contents by replacing the binary visiable layer with Gaussian visiable layer.</br>
</br>

## **Instruction**</br>
All the codes in the repository are written in Python 2.7.</br>
To simplify the installation of Python 2.7, I use Anaconda 2.4.1.</br>
The dependencies are [GNU Scientific Library (GSL ver. 1.14)](https://www.gnu.org/software/gsl/), [theano (ver 0.8)] (http://deeplearning.net/software/theano/).</br>
After forking, you should configure several things before running the code:</br>
- Download GSL and Anaconda then install, use pip to install theano;
- Enter directory 'cr', modify the Makefile to configure the library path of GSL, and compile;
- Create the directories for the upcoming models. </br>

If you are OK with using the project directory as workspace, please run intialize.sh:</br>
```
chmod +x initialize.sh
sh initialize.sh
```
Enter directory 'methods' to run the training script of CER and DPM.</br>
You can use following commands (let me use content 'tfidf' as example):
```
python clr_train.py -fp ../contents/tfidf.npy -fn tfidf -wd ../models/cer
python dpm_train.py -fp ../contents/tfidf.npy -fn tfidf -wd ../models/dpm
```
To evaluate CER and DPM, setting the variable 'model_root' (default is '../models/cer') in test.py, then run:
```
python test.py
```
To train BPR and VBPR, you can use following commands:
```
python bpr_train.py
python vbpr_train.py
```
To evaluate BPR and VBPR, you can use following commands:
```
python bpr_test.py
python vbpr_test.py
```
To evaluate the fusion methods, you can use following commands:
```
python pfusion.py // the proposed fusion method
python afusion.py // the average fusion method
python bfusion.py // the bpr based fusion method
python efusion.py // the error based fusion method, the error is measured by RMSE
python sfusion.py // the svm based fusion method
```
## **Dataset**</br>
Due to the file size limitation, the data for training and testing are maintained by other services.</br>
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
