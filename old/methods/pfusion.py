"""
    The code for performing fusion and testing
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
"""

import cPickle as pickle
import numpy as np
import os, sys

model_root = '../models/cer'
uidpath  = '../complete/uid';
vidpath  = '../complete/vid';
tridpath = '../complete/f%dtr.idl';
trpath   = '../complete/f%dtr.txt';
teidpath = '../complete/f%dte.%s.idl';
tepath   = '../complete/f%dte.%s.txt'
scenario = ['im', 'om'];
start     = 0;
end       = 5;
total     = 30;
step      = 5;
interval  = total / step;
feats     = ['tfidf', 'meta', 'cnnfv', 'cnnvlad', 'idt', 'mfcc', 'osift', 'mosift'];
upath     = os.path.join(model_root, '/%s%d/final-U.dat');
vpath     = os.path.join(model_root, '/%s%d/final-V.dat');
featset1  = {'meta':0, 'tfidf':1, 'cnnfv':2, 'cnnvlad':3, 'idt':4, 'mfcc':5, 'osift':6, 'mosift':7};
featset2  = {'cnnfv':0, 'cnnvlad':1, 'idt':2, 'mfcc':3, 'osift':4, 'mosift':5};

def getU(fpath, uids):
    umatrix = None;
    lines = open(fpath).readlines();
    for uid in uids:
        terms = lines[uids[uid]].strip().split(' ');
        if umatrix is None:
            umatrix = np.zeros((len(uids), len(terms)), dtype=np.float32);
        for k in range(len(terms)):
            umatrix[uids[uid], k] = np.float32(terms[k]);
    return umatrix;

def getV(fpath, vids):
    vmatrix = None;
    lines = open(fpath).readlines();
    for vid in vids:
        terms = lines[vids[vid]].strip().split(' ');
        if vmatrix is None:
            vmatrix = np.zeros((len(vids), len(terms)), dtype=np.float32);
        for k in range(len(terms)):
            vmatrix[vids[vid], k] = np.float32(terms[k]);
    return vmatrix;

def doScore(upath, vpath, uids, vids, tevids):
    U    = getU(upath, uids);
    V    = getV(vpath, vids);
    teV  = np.zeros((len(tevids), V.shape[1]), dtype=np.float32);
    for vid in vids:
        if vid in tevids:
            teV[tevids[vid],:] = V[vids[vid],:];
    scores = np.dot(U, teV.T);
    return scores;

def doRankFusion(scores, p):
    custom_weights = np.zeros(scores.shape[2], dtype = np.float32);
    fusion         = np.zeros(scores[:,:,0].shape, dtype=np.float32);
    for i in range(scores.shape[2]):
        custom_weights[i] = np.power(1 - p, i) * p;
    for i in range(scores.shape[2]):
        fusion += custom_weights[i] * scores[:,:,i];
    return fusion;

def doEvaluate(score, tepath, users, teiids, invids):
    count   = 0;
    results = [0.0] * interval;
    ranks   = np.argsort(score, axis=1);
    for line in open(tepath):
        terms = line.strip().split(',');
        uid   = int(terms[0]);
        likes = list();
        for i in range(1, len(terms)):
            iid  = int(terms[i].split(':')[0]);
            like = int(terms[i].split(':')[1]);
            if like == 1:
                likes.append(iid);
        if len(likes) != 0 and uid in users:
            hits = [0] * interval;
            for t in range(total):
                if invids[ranks[users[uid], len(teiids)-1-t]] in likes:
                    j = t / step;
                    for k in range(j, interval):
                        hits[k] += 1;
            for k in range(interval):
                results[k] += hits[k];
            count += len(likes);
    return results, count;
    
def get_all_ids(idpath):
    ids = dict();
    for line in open(idpath):
        tid = int(line.strip());
        ids[tid] = len(ids);
    return ids;

def get_test_vids(teidpath):
    teiids = dict();
    invids = dict();
    for line in open(teidpath):
        iid = int(line.strip());
        teiids[iid] = len(teiids);
        invids[len(invids)] = iid;
    return teiids, invids;

def main():
    ps = np.arange(0.1,1.0,0.1);
    results = {'im':dict(),'om':dict()};
    vids = get_all_ids(vidpath);
    uids = get_all_ids(uidpath);
    for i in range(start, end):
        for mode in scenario:
            tevids, invids = get_test_vids(teidpath%(i, mode));
            scores1 = None;
            scores2 = None;
            users   = None;
            for fname in feats:
                if fname in featset1 or fname in featset2:
                    score = None;
                    if scores1 is None:
                        scores1 = np.zeros((len(uids), len(tevids), len(featset1)), dtype=np.float32)
                    if scores2 is None:
                        scores2 = np.zeros((len(uids), len(tevids), len(featset2)), dtype=np.float32)
                    score = doScore(upath%(fname,i), vpath%(fname,i), uids, vids, tevids);
                    if fname in featset1:
                        scores1[:,:,featset1[fname]] = score;
                    if fname in featset2:
                        scores2[:,:,featset2[fname]] = score;
            for p in ps:
                rfusion = doRankFusion(scores1, p);
                tresults, tcount = doEvaluate(rfusion, tepath%(i, mode), uids, tevids, invids);
                for j in range(interval):
                    tresults[j] = tresults[j] * 1.0 / tcount;
                if 'pfusion1' not in results[mode]:
                    results[mode]['pfusion1'] = dict();
                if p not in results[mode]['pfusion1']:
                    results[mode]['pfusion1'][p] = np.array(tresults);
                else:
                    for j in range(interval):
                        results[mode]['pfusion1'][p][j] += tresults[j];
                rfusion = doRankFusion(scores2, p);
                tresults, tcount = doEvaluate(rfusion, tepath%(i, mode), uids, tevids, invids);
                for j in range(interval):
                    tresults[j] = tresults[j] * 1.0 / tcount;
                if 'pfusion2' not in results[mode]:
                    results[mode]['pfusion2'] = dict();
                if p not in results[mode]['pfusion2']:
                    results[mode]['pfusion2'][p] = np.array(tresults);
                else:
                    for j in range(interval):
                        results[mode]['pfusion2'][p][j] += tresults[j];

    for mode in scenario:
        print '%s results:'%mode
        for title in results[mode]:
            print title;
            for p in results[mode][title]:
                line = str(p);
                for j in range(interval):
                    results[mode][title][p][j] /= (end-start);
                    line += ',%.9f'%results[mode][title][p][j];
                print line;

if __name__ == '__main__':
    main();
