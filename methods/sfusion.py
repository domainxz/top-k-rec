import cPickle as pickle
import numpy as np
import os, sys
from sklearn import svm
from efusion import getU, getV, doScore, doEvaluate, get_all_ids, get_test_vids

model_root = '../models/cer';
uid = '../complete/uid';
vid = '../complete/vid';
tridpath = '../complete/f%dtr.idl';
trpath = '../complete/f%dtr.txt';
teidpath = '../complete/f%dte.%s.idl';
tepath = '../complete/f%dte.%s.txt'
scenario = ['im', 'om'];
iteration = 5;
start     = 0;
end       = 5;
total     = 30;
step      = 5;
nsample   = 1000000;
interval  = total / step;
feats     = ['tfidf', 'meta', 'cnnfv', 'cnnvlad', 'idt', 'mfcc', 'osift', 'mosift'];
upath  = os.path.join(model_root, '/%s%d/final-U.dat');
vpath  = os.path.join(model_root, '/%s%d/final-V.dat');
featset1  = {'meta':0, 'tfidf':1, 'cnnfv':2, 'cnnvlad':3, 'idt':4, 'mfcc':5, 'osift':6, 'mosift':7};
featset2  = {'cnnfv':0, 'cnnvlad':1, 'idt':2, 'mfcc':3, 'osift':4, 'mosift':5};

def get_weights(trscores, uids, trvids, i):
    clf = svm.LinearSVC(C=0.01);
    usm = dict();
    x   = list();
    y   = list();
    for line in open(trpath%i):
        terms = line.strip().split(',');
        uid   = int(terms[0]);
        likes = list();
        for j in range(1, len(terms)):
            vid  = int(terms[j].split(':')[0]);
            like = int(terms[j].split(':')[1]);
            if like == 1:
                likes.append(vid);
        if len(likes) != 0:
            usm[uid] = likes;
    sgd_users = np.array(list(usm.keys()))[np.random.randint(len(usm), size=nsample)];
    x = np.zeros((nsample, trscores.shape[2]), dtype=np.float32);
    y = np.zeros(nsample, dtype=np.int8);
    for k in range(nsample):
        uid = sgd_users[k];
        lvid = usm[uid][0];
        if len(usm[uid]) > 1:
            lvid = usm[uid][np.random.randint(len(usm[uid]))];
        dvid = trvids.keys()[np.random.randint(len(trvids))];
        while dvid in usm[uid]:
            dvid = trvids.keys()[np.random.randint(len(trvids))];
        feat = trscores[uids[uid], trvids[lvid], :] - trscores[uids[uid], trvids[dvid], :];
        if k % 2 == 0:
            x[k] = feat;
            y[k] = 1;
        else:
            x[k] = -feat;
            y[k] = -1;
    clf.fit(x, y);
    return clf.coef_[0];

def do_fusion(weights, tescores):
    fusion = np.zeros(tescores[:,:,0].shape, dtype=np.float32);
    for j in range(tescores.shape[2]):
        fusion += weights[j] * tescores[:, :, j];
    return fusion

def main():
    results = {'im':dict(),'om':dict()};
    uids = get_all_ids(uid);
    vids = get_all_ids(vid);
    for i in range(start, end):
        trvids = get_all_ids(tridpath%(i));
        trscores1 = np.zeros((len(uids), len(trvids), len(featset1)), dtype=np.float32);
        trscores2 = np.zeros((len(uids), len(trvids), len(featset2)), dtype=np.float32);
        for fname in feats:
            if fname in featset1 or fname in featset2:
                trscore = doScore(upath%(fname,i), vpath%(fname,i), uids, vids, trvids);
                if fname in featset1:
                    trscores1[:,:,featset1[fname]] = trscore;
                if fname in featset2:
                    trscores2[:,:,featset2[fname]] = trscore;
        weights1 = get_weights(trscores1, uids, trvids, i);
        weights2 = get_weights(trscores2, uids, trvids, i);
        for mode in scenario:
            tevids, invids = get_test_vids(teidpath%(i, mode));
            tescores1 = np.zeros((len(uids), len(tevids), len(featset1)), dtype=np.float32);
            tescores2 = np.zeros((len(uids), len(tevids), len(featset2)), dtype=np.float32);
            for fname in feats:
                if fname in featset1 or fname in featset2:
                    tescore = doScore(upath%(fname,i), vpath%(fname,i), uids, vids, tevids);
                    if fname in featset1:
                        tescores1[:,:,featset1[fname]] = tescore;
                    if fname in featset2:
                        tescores2[:,:,featset2[fname]] = tescore;
            sfusion = do_fusion(weights1, tescores1);
            tresults, tcount = doEvaluate(sfusion, tepath%(i, mode), uids, tevids, invids);
            for j in range(interval):
                tresults[j] = tresults[j] * 1.0 / tcount;
            if 'sfusion1' not in results[mode]:
                results[mode]['sfusion1'] = np.array(tresults);
            else:
                for j in range(interval):
                    results[mode]['sfusion1'][j] += tresults[j];
            sfusion = do_fusion(weights2, tescores2); 
            tresults, tcount = doEvaluate(sfusion, tepath%(i, mode), uids, tevids, invids);
            for j in range(interval):
                tresults[j] = tresults[j] * 1.0 / tcount;
            if 'sfusion2' not in results[mode]:
                results[mode]['sfusion2'] = np.array(tresults);
            else:
                for j in range(interval):
                    results[mode]['sfusion2'][j] += tresults[j];

    for mode in scenario:
        print '%s results:'%mode
        for title in results[mode]:
            line = title;
            for j in range(interval):
                results[mode][title][j] /= (end-start);
                line += ',%.9f'%results[mode][title][j];
            print line;

if __name__ == '__main__':
    main();
