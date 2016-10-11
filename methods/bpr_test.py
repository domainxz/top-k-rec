import cPickle as pickle
import numpy as np
import theano.tensor as T
import os, sys

model_root = '';
vid = '../complete/vid';
teidpath = '../complete/f%dte.%s.idl';
tepath = '../complete/f%dte.%s.txt'
trpath = '../complete/f%dtr.txt'
scenario = ['im'];
fold      = 5;
total     = 30;
step      = 5;
interval  = total / step;
modelpath = os.path.join(model_root, '/%d.bpr');

def doRank(iids, teiids, model, mode):
    H = np.array(model.H.eval());
    h = np.zeros((len(teiids), H.shape[1]), dtype=np.float32);
    B = np.array(model.B.eval());
    b = np.zeros(len(teiids), dtype=np.float32);
    for iid in iids:
        if iid in teiids:
            if H is not None and B is not None and iid in model._train_items:
                h[teiids[iid],:] = H[model._train_items[iid],:];
                b[teiids[iid]]   = B[model._train_items[iid]];
    bias    = np.zeros(len(teiids), dtype=np.float32);
    bias   += b;
    scores  = np.tile(bias.reshape((1,len(teiids))), (model._n_users, 1));
    scores += np.array(T.dot(model.W, h.T).eval());
    return scores;

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
    results = {'im':[0.0] * interval};
    iids = get_all_vids(vid);
    for i in range(fold):
        rated, popular = get_rated_and_popular(trpath%i);
        model = pickle.load(open(modelpath%(i), 'rb'));
        for mode in scenario:
            teiids, invids = get_test_vids(teidpath%(i, mode));
            scores = doRank(iids, teiids, model, mode);
            tresults, tcount = doEvaluate(scores, model, rated, popular, tepath%(i, mode), teiids, invids);
            for j in range(interval):
                tresults[j] = tresults[j] * 1.0 / tcount;
                results[mode][j] += tresults[j];
    for mode in scenario:
        print '%s results:'%mode
        line = 'none';
        for j in range(interval):
            results[mode][j] /= fold;
            line += ',%.9f'%results[mode][j];
        print line;

if __name__ == '__main__':
    main();
