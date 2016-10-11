"""
    The code for testing the proposed CER method, or DPM method
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
"""

import numpy as np
import cPickle as pickle
from sklearn import metrics
import os

model_root = '';
fold  = 5;
scenarios = ['im', 'om'];
feats     = ['meta','tfidf','mfcc','idt','cnnfv','cnnvlad','mosift','osift'];
step      = 5;
total     = 30;
interval  = total / step;
results   = dict();
umap  = dict();
vmap  = dict();
for line in open('../complete/uid'):
    uid = int(line.strip());
    umap[uid] = len(umap);
for line in open('../complete/vid'):
    vid = int(line.strip());
    vmap[vid] = len(vmap);
for i in range(fold):
    trids = dict();
    umatrix = None;
    rated = dict();
    popular = dict();
    for line in open('../complete/f%dtr.idl'%i):
        vid = int(line.strip());
        trids[vid] = len(trids);
    for line in open('../complete/f%dtr.txt'%i):
        terms = line.strip().split(',');
        uid = int(terms[0]);
        rated[uid] = set();
        for k in range(1, len(terms)):
            vid = int(terms[k].split(':')[0]);
            like = int(terms[k].split(':')[1]);
            if vid in trids:
                rated[uid].add(trids[vid]);
                if like == 1:
                    if trids[vid] not in popular:
                        popular[trids[vid]] = 1;
                    else:
                        popular[trids[vid]] += 1;
    flag = int(len(popular) * 0.06);
    popitems = sorted(popular, key=popular.get, reverse=True)[0:flag];
    for feat in feats:
        lines = open(os.path.join(model_root, '/%s%d/final-U.dat'%(feat, i))).readlines();
        for uid in umap:
            terms = lines[umap[uid]].strip().split(' ');
            if umatrix is None:
                umatrix = np.zeros((len(umap), len(terms)), dtype=np.float32);
            for k in range(len(terms)):
                umatrix[umap[uid], k] = np.float32(terms[k]);
        for scenario in scenarios:
            teids = dict();
            imatrix = None;
            for line in open('../complete/f%dte.%s.idl'%(i, scenario)):
                vid = int(line.strip());
                teids[vid] = len(teids);
            lines = open(os.path.join(model_root, '/%s%d/final-V.dat'%(feat, i))).readlines();
            for vid in teids:
                terms = lines[vmap[vid]].strip().split(' ');
                if imatrix is None:
                    imatrix = np.zeros((len(teids), len(terms)), dtype=np.float32);
                for k in range(len(terms)):
                    imatrix[teids[vid], k] = np.float32(terms[k]);
            scores = np.dot(umatrix, imatrix.T);
            rlist  = np.argsort(scores, axis=1);
            tresults = [0.0]*interval;
            tcount = 0;
            for line in open('../complete/f%dte.%s.txt'%(i, scenario)):
                terms = line.strip().split(',');
                uid   = int(terms[0]);
                likes = set();
                idx   = 0;
                for k in range(1, len(terms)):
                    vid = int(terms[k].split(':')[0]);
                    like = int(terms[k].split(':')[1]);
                    if like == 1:
                        likes.add(teids[vid]);
                if len(likes) != 0:
                    hits = [0] * interval;
                    for t in range(len(teids)):
                        liid = rlist[umap[uid], len(teids)-t-1];
                        if liid not in rated[uid]:
                            if liid in likes:
                                j = idx / step;
                                for k in range(j, interval):
                                    hits[k] += 1;
                            idx += 1;
                        if idx == total:
                            break;
                    for k in range(interval):
                        tresults[k] += hits[k];
                    tcount += len(likes);
            if scenario not in results:
                results[scenario] = dict();
            if feat not in results[scenario]:
                results[scenario][feat] = [0.0]*interval;
            for k in range(interval):
                results[scenario][feat][k] += tresults[k] / tcount;
for scenario in scenarios:
    print scenario;
    for feat in feats:
        line = feat;
        for k in range(interval):
            line += ',%f'%(results[scenario][feat][k] / fold);
        print line;
