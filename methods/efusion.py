import cPickle as pickle
import numpy as np
import os, sys
import scipy.sparse as ss

model_root = '../models/cer'
uidpath = '../complete/uid';
vidpath = '../complete/vid';
tridpath = '../complete/f%dtr.idl';
trpath = '../complete/f%dtr.txt';
teidpath = '../complete/f%dte.%s.idl';
tepath = '../complete/f%dte.%s.txt'
scenario = ['im', 'om'];
start     = 0;
end       = 5;
total     = 30;
step      = 5;
interval  = total / step;
feats     = ['tfidf', 'meta', 'cnnfv', 'cnnvlad', 'idt', 'mfcc', 'osift', 'mosift'];
upath  = os.path.join(model_root, '/%s%d/final-U.dat');
vpath  = os.path.join(model_root, '/%s%d/final-V.dat');
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

def get_weights(trscores, uids, trvids, i, featset):
    weight = np.zeros((trscores.shape[0], len(featset)), dtype=np.float32);
    row    = list();
    col    = list();
    rat    = list();
    for line in open(trpath%i):
        terms = line.strip().split(',');
        uid = int(terms[0]);
        for j in range(1, len(terms)):
            vid  = int(terms[j].split(':')[0]);
            like = int(terms[j].split(':')[1]);
            if like == 1:
                row.append(uids[uid]);
                col.append(trvids[vid]);
                rat.append(1.0);
    lmat = ss.csc_matrix((rat, (row, col)), shape=(len(uids), len(trvids)), dtype=np.float32).toarray();
    svec = np.sum(lmat,axis=1);
    svec[svec==0] = 1;
    for feat in featset:
        weight[:, featset[feat]] = np.sqrt(np.divide(np.sum(np.multiply((trscores[:,:,featset[feat]] - lmat) ** 2, lmat), axis=1), svec));
    for j in range(weight.shape[0]):
        wmean = np.mean(weight[j,:]);
        if wmean != 0:
            weight[j,:] -= wmean;
            weight[j,:] = np.exp(-weight[j,:]);
    return weight;

def do_fusion(weights, tescores):
    fusion = np.zeros(tescores[:,:,0].shape, dtype=np.float32);
    for j in range(tescores.shape[0]):
        for k in range(tescores.shape[2]):
            fusion[j,:] += weights[j,k] * tescores[j,:,k];
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
    results = {'im':dict(),'om':dict()};
    vids = get_all_ids(vidpath);
    uids = get_all_ids(uidpath);
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
        weights1 = get_weights(trscores1, uids, trvids, i, featset1);
        weights2 = get_weights(trscores2, uids, trvids, i, featset2);
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
            rfusion = do_fusion(weights1, tescores1);
            tresults, tcount = doEvaluate(rfusion, tepath%(i, mode), uids, tevids, invids);
            for j in range(interval):
                tresults[j] = tresults[j] * 1.0 / tcount;
            if 'efusion1' not in results[mode]:
                results[mode]['efusion1'] = np.array(tresults);
            else:
                for j in range(interval):
                    results[mode]['efusion1'][j] += tresults[j];
            rfusion = do_fusion(weights2, tescores2);
            tresults, tcount = doEvaluate(rfusion, tepath%(i, mode), uids, tevids, invids);
            for j in range(interval):
                tresults[j] = tresults[j] * 1.0 / tcount;
            if 'efusion2' not in results[mode]:
                results[mode]['efusion2'] = np.array(tresults);
            else:
                for j in range(interval):
                    results[mode]['efusion2'][j] += tresults[j];

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
