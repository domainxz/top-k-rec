import cPickle as pickle
import numpy as np
import os, sys
from efusion import getU, getV, doScore, doEvaluate, get_all_ids, get_test_vids

model_root = '../models/cer'
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
interval  = total / step;
feats     = ['tfidf', 'meta', 'cnnfv', 'cnnvlad', 'idt', 'mfcc', 'osift', 'mosift'];
upath  = os.path.join(model_root, '/%s%d/final-U.dat');
vpath  = os.path.join(model_root, '/%s%d/final-V.dat');
featset1  = {'meta':0, 'tfidf':1, 'cnnfv':2, 'cnnvlad':3, 'idt':4, 'mfcc':5, 'osift':6, 'mosift':7};
featset2  = {'cnnfv':0, 'cnnvlad':1, 'idt':2, 'mfcc':3, 'osift':4, 'mosift':5};

def do_fusion(tescores):
    weight = 1.0 / float(tescores.shape[2]);
    fusion = np.zeros(tescores[:,:,0].shape, dtype=np.float32);
    for j in range(tescores.shape[2]):
        fusion += weight * tescores[:, :, j];
    return fusion

def main():
    results = {'im':dict(),'om':dict()};
    uids = get_all_ids(uid);
    vids = get_all_ids(vid);
    for i in range(start, end):
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
            afusion = do_fusion(tescores1);
            tresults, tcount = doEvaluate(afusion, tepath%(i, mode), uids, tevids, invids);
            for j in range(interval):
                tresults[j] = tresults[j] * 1.0 / tcount;
            if 'afusion1' not in results[mode]:
                results[mode]['afusion1'] = np.array(tresults);
            else:
                for j in range(interval):
                    results[mode]['afusion1'][j] += tresults[j];
            afusion = do_fusion(tescores2); 
            tresults, tcount = doEvaluate(afusion, tepath%(i, mode), uids, tevids, invids);
            for j in range(interval):
                tresults[j] = tresults[j] * 1.0 / tcount;
            if 'afusion2' not in results[mode]:
                results[mode]['afusion2'] = np.array(tresults);
            else:
                for j in range(interval):
                    results[mode]['afusion2'][j] += tresults[j];

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
