import cPickle as pickle
import numpy as np
import theano.tensor as T
from vbpr import VBPR
import scipy.sparse as ss
import os, sys

model_root = '../models/vbpr';
vid = '../complete/vid';
teidpath = '../complete/f%dte.%s.idl';
tepath = '../complete/f%dte.%s.txt'
trpath = '../complete/f%dtr.txt'
scenario = ['im', 'om'];
fold      = 5;
total     = 30;
step      = 5;
interval  = total / step;
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

def doRank(featpath, iids, teiids, model, mode):
    allfeat = pickle.load(open(featpath, 'rb'));
    if ss.issparse(allfeat):
        allfeat = allfeat.toarray();
    tefeat  = np.zeros((len(teiids), allfeat.shape[1]), dtype=np.float32);
    H = np.array(model.H.eval());
    h = np.zeros((len(teiids), H.shape[1]), dtype=np.float32);
    B = np.array(model.B.eval());
    b = np.zeros(len(teiids), dtype=np.float32);
    for iid in iids:
        if iid in teiids:
            tefeat[teiids[iid],:] = allfeat[iids[iid],:];
            if H is not None and B is not None and iid in model._train_items:
                h[teiids[iid],:] = H[model._train_items[iid],:];
                b[teiids[iid]]   = B[model._train_items[iid]];
    bias   = np.zeros(tefeat.shape[0], dtype=np.float32);
    if mode == 'im':
        bias += b;
    bias  += np.array(T.dot(model.C, tefeat.T).eval());
    scores = np.tile(bias.reshape((1,len(teiids))), (model._n_users, 1));
    if mode == 'im':
        scores += np.array(T.dot(model.W, h.T).eval());
    scores += np.array(T.dot(model.P, T.dot(model.E, tefeat.T)).eval());
    return scores;
    ranks  = np.argsort(scores, axis=1);
    return ranks;

def doEvaluate(scores, model, rated, popular, tepath, teiids, invids):
    ranks  = np.argsort(scores, axis=1);
    count   = 0;
    results = [0.0] * interval;
    for line in open(tepath):
        terms = line.strip().split(',');
        uid   = int(terms[0]);
        likes = set();
        idx = 0;
        for i in range(1, len(terms)):
            iid  = int(terms[i].split(':')[0]);
            like = int(terms[i].split(':')[1]);
            if like == 1:
                likes.add(iid);
        if len(likes) != 0 and uid in model._train_users:
            hits = [0] * interval;
            for t in range(len(teiids)):
                liid = invids[ranks[model._train_users[uid], len(teiids)-1-t]];
                if liid not in rated[uid]:
                    if liid in likes:
                        j = idx / step;
                        for k in range(j, interval):
                            hits[k] += 1;
                    idx += 1;
                if idx == total:
                    break;
            for k in range(interval):
                results[k] += hits[k];
            count += len(likes);
    return results, count;
    
def get_all_vids(vidpath):
    iids = dict();
    for line in open(vidpath):
        iid = int(line.strip());
        iids[iid] = len(iids);
    return iids;

def get_test_vids(teidpath):
    teiids = dict();
    invids = dict();
    for line in open(teidpath):
        iid = int(line.strip());
        teiids[iid] = len(teiids);
        invids[len(invids)] = iid;
    return teiids, invids;

def get_rated_and_popular(trpath):
    rated = dict();
    popularity = dict();
    for line in open(trpath):
        terms = line.strip().split(',');
        uid = int(terms[0]);
        rated[uid] = set();
        for i in range(1, len(terms)):
            iid = int(terms[i].split(':')[0]);
            like = int(terms[i].split(':')[1]);
            rated[uid].add(iid);
            if like == 1:
                if iid not in popularity:
                    popularity[iid] = 1;
                else:
                    popularity[iid] += 1;
    flag = int(len(popularity) * 0.06);
    popitems = sorted(popularity, key=popularity.get, reverse=True)[0:flag];
    return rated, popitems;

def main():
    results = {'im':dict(),'om':dict()};
    iids = get_all_vids(vid);
    for i in range(fold):
        rated, popular = get_rated_and_popular(trpath%i);
        for fname in fpath:
            model = pickle.load(open(modelpath%(fname,i), 'rb'));
            for mode in scenario:
                teiids, invids = get_test_vids(teidpath%(i, mode));
                scores = doRank(fpath[fname], iids, teiids, model, mode);
                tresults, tcount = doEvaluate(scores, model, rated, popular, tepath%(i, mode), teiids, invids);
                for j in range(interval):
                    tresults[j] = tresults[j] * 1.0 / tcount;
                if fname not in results[mode]:
                    results[mode][fname] = tresults;
                else:
                    for j in range(interval):
                        results[mode][fname][j] += tresults[j];
    for mode in scenario:
        print '%s results:'%mode
        for fname in fpath:
            line = fname;
            for j in range(interval):
                results[mode][fname][j] /= fold;
                line += ',%.9f'%results[mode][fname][j];
            print line;

if __name__ == '__main__':
    main();
