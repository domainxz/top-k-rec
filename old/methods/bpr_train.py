import cPickle as pickle
import numpy as np
from bpr import BPR
import os, sys

model_root = '../models/bpr'
vid = '../complete/vid';
tridpath = '../complete/f%dtr.idl';
trpath = '../complete/f%dtr.txt';
dimension = 50;
iteration = 5;
fold      = 5;
modelpath  = os.path.join(model_root, '/%d.bpr');

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
    trfeat  = np.zeros((len(triids), allfeat.shape[1]), dtype=np.float32);
    for iid in iids:
        if iid in triids:
            trfeat[triids[iid],:] = allfeat[iids[iid],:];
    return trfeat;

def main():
    iids = get_all_vids(vid);
    for i in range(fold):
        data, truids, triids = get_train_dicts(trpath%i);
        model = BPR(dimension, truids, triids);
        model.train(data, iteration);
        pickle.dump(model, open(modelpath%(i), 'wb'), pickle.HIGHEST_PROTOCOL);

if __name__ == '__main__':
    main();
