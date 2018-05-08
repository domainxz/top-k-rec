import cPickle as pickle
import numpy as np
from vbpr import VBPR
import os, sys
import scipy.sparse as ss

model_root = '../models/vbpr';
vid = '../complete/vid';
tridpath = '../complete/f%dtr.idl';
trpath = '../complete/f%dtr.txt';
teidpath = '../complete/f%dte.%s.idl';
tepath = '../complete/f%dte.%s.txt'
scenario = ['im', 'om'];
dimension = 25;
iteration = 5;
fold      = 5;
fpath = {
          'meta':'../contents/meta.npy',
          'tfidf':'../contents/tfidf.npy',
          'cnnfv':'../contents/cnn.fv.trans',
          'cnnvlad':'../contents/cnn.vlad.trans',
          'idt':'../contents/idt.trans',
          'mosift':'../contents/mosift.trans',
          'osift':'../contents/osift.trans',
          'mfcc':'../contents/mfcc.trans'
         }
modelpath = os.path.join(model_root, '/%s.%d.vbpr');

def get_all_vids(vidpath):
    iids = dict();
    for line in open(vidpath):
        iid = int(line.strip());
        iids[iid] = len(iids);
    return iids;

def get_train_dicts(trpath):
    data   = list();
    truids = dict();
    triids = dict();
    for line in open(os.path.join(trpath)):
        terms = line.strip().split(',');
        uid = int(terms[0]);
        if len(terms) > 1:
            truids[uid] = len(truids);
            for j in range(1,len(terms)):
                iid  = int(terms[j].split(':')[0]);
                like = int(terms[j].split(':')[1]);
                if iid not in triids:
                    triids[iid] = len(triids);
                if like == 1:
                    data.append((uid, iid));
    return data, truids, triids

def get_train_features(feapath, triids, iids):
    allfeat = pickle.load(open(feapath));
    if ss.issparse(allfeat):
        allfeat = allfeat.toarray();
    trfeat  = np.zeros((len(triids), allfeat.shape[1]), dtype=np.float32);
    for iid in iids:
        if iid in triids:
            trfeat[triids[iid],:] = allfeat[iids[iid],:];
    return trfeat;

def get_initW(wpath):
    W = None;
    lines = open(wpath).readlines();
    for i in range(len(lines)):
        terms = lines[i].strip().split(' ');
        if W is None:
            W = np.zeros((len(lines), len(terms)), dtype=np.float32);
        for j in range(len(terms)):
            W[i, j] = np.float32(terms[j]);
    return W;

def main():
    iids = get_all_vids(vid);
    for i in range(fold):
        data, truids, triids = get_train_dicts(trpath%i);
        for fname in fpath:
            content = get_train_features(fpath[fname], triids, iids);
            model = VBPR(dimension, content, truids, triids);
            model.train(data, iteration);
            pickle.dump(model, open(modelpath%(fname,i), 'wb'), pickle.HIGHEST_PROTOCOL);

if __name__ == '__main__':
    main();
