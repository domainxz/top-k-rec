# Top-k Recommendation
## **Introduction**</br>

A python code collection for top-k recommendation refactored by tensorflow</br>
The old repository is [here](old).</br>
Current implementation is purely based on python, however, its speed is slower than the old one.

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

For training, you can run
```
python train.py
```
For evaluation, you can run
```
python evaluate.py -d data -m embed/cer -f 0 -sl im om
```
This will evaluate cer's performance in both in-matrix and out-of-matrix settings with content feature (In our example, this is meta).<br>
By default, the evaluation will report accuracy@5,10,15,20,25 and 30.<br>

## **Dataset**</br>
Due to the file size limitation, datasets for training and testing are hosted by other places.</br>
At present, we provide two datasets derived from Movielens 10M and Netflix:<br>
Movielens: <a href="https://drive.google.com/file/d/1nMhFTlWMEol9kbWx6SOQX_FF8IcK0WoI/view?usp=sharing">ratings</a> and <a href="https://drive.google.com/file/d/1FPhBhunJxLpULb_4JkjIiA1-0p-IruI1/view?usp=sharing">features</a><br>
Netflix: <a href="https://drive.google.com/file/d/1VDfPeBfg2PpCIbKsQq6upgyjRN-asY-R/view?usp=sharing">rating</a> and <a href="https://drive.google.com/file/d/1O_76Wt6wblJkm3JYohU3X1hwH8uDziE-/view?usp=sharing">features</a><br>
Each of them will have following data files for experiments:<br>
  - uid: 
      - User id list where each line is a user id. The id sequence may not be continuous.
  - vid: 
      - Video id list where each line is a video id. The id sequence may not be continuous.
  - f?[tr|te][.|.im|.om].[idl|txt]:
      - Rating related files where ? is the fold index, tr denotes training set, te denotes testing set, im denotes in-matrix evaluation, om indicate out-of-matrix evaluation, idl denotes id list and txt denotes rating file.
      - Each line in rating files starts with a used id, and is filled with the corresponding item-rating pairs separated by commas. In each video-rating pair, 1 denotes like and 0 denotes dislike.
      - For instance:
        1. f2tr.txt contains the ratings in the training set 2
        2. f2te.im.txt contains the ratings in the test set 2 for in-matrix evaluation
        3. f2te.om.txt contains the ratings in the test set 2 for out-matrix evaluation
  - The input data files for ctr are also provided. Their suffixes are 'mfp'.
  - The feature files could be read by pickle in binary mode. The feature vectors are aligned to the id list in vid.

Please modify the access path inside code to make the execution correctly.</br>

The original 10380 videos can be downloaded from below link:</br>
<a href="https://drive.google.com/drive/folders/1hK7WgQOqllsozB9oeVmbMPQtZ0w3rf9b?usp=sharing">Google Drive</a></br>
The video meta information such as title, plot and actors are in imdbpy.tgz.</br>
The meta information uses <a href="https://imdbpy.github.io/">imdbpy</a>. </br>
Please install it first and use pickle to read provided files in the binary mode. </br>
For instance, you can access the imdbpy object for video 999 by:
```
>>> import imdb
>>> import pickle
>>> meta_999 = pickle.load(open('999.pkl', 'rb'))
```
## **Reference**</br>
If you use above codes or data, please cite the paper below:</br>
@article{VCRS, </br>
&nbsp;&nbsp;&nbsp;&nbsp;author    = {Xingzhong Du and Hongzhi Yin and Ling Chen and Yang Wang and Yi Yang and Xiaofang Zhou}, </br>
&nbsp;&nbsp;&nbsp;&nbsp;title     = {Personalized Video Recommendation Using Rich Contents from Videos}, </br>
&nbsp;&nbsp;&nbsp;&nbsp;journal   = {TKDE}, </br>
&nbsp;&nbsp;&nbsp;&nbsp;year      = {2019} </br>
} </br>
