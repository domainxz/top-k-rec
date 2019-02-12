import argparse
import numpy as np
import os

def get_ids(idpath):
    ids = dict();
    for line in open(idpath):
        tid = line.strip();
        ids[tid] = len(ids);
    return ids;

def get_ivt(idpath):
    ivt = dict();
    for line in open(idpath):
        tid = line.strip();
        ivt[len(ivt)] = tid;
    return ivt;

def get_mat(mpath, mids):
    mat = None;
    lines = open(mpath).readlines();
    for mid in mids:
        terms = lines[mids[mid]].strip().split(' ');
        if mat is None:
            mat = np.zeros((len(mids), len(terms)), dtype=np.float32);
        for k in range(len(terms)):
            mat[mids[mid], k] = np.float32(terms[k]);
    return mat;

def get_history(hpath):
    rated = dict();
    popular = dict();
    for line in open(hpath):
        terms = line.strip().split(',');
        uid   = terms[0];
        rated[uid] = set();
        for k in range(1, len(terms)):
            vid  = terms[k].split(':')[0];
            like = int(terms[k].split(':')[1]);
            rated[uid].add(vid);
            if like == 1:
                if vid not in popular:
                    popular[vid] = 0;
                popular[vid] += 1;
    return rated, popular;

def main():
    parser = argparse.ArgumentParser(description="Evaluate weighted matrix factorization based methods.")
    parser.add_argument('-d',  '--data',      required=True,               help="The data path for the evaluation");
    parser.add_argument('-m',  '--model',     required=True,               help="The work path for the model");
    parser.add_argument('-f',  '--fold',      type=int,      default=5,    help="The number of evaluation fold");
    parser.add_argument('-s',  '--step',      type=int,      default=5,    help="The number of evaluation step");
    parser.add_argument('-t',  '--total',     type=int,      default=30,   help="The number of total predictions");
    parser.add_argument('-sl', '--scenarios', nargs='+',     default=None, help="The test scenario list");
    parser.add_argument('-fl', '--features',  nargs='+',     default=None, help="The feature list");
    args = parser.parse_args();
    
    uids = get_ids(os.path.join(args.data, 'uid'));
    vids = get_ids(os.path.join(args.data, 'vid'));
    fold = args.fold;
    scenarios = args.scenarios;
    features  = args.features;
    step      = args.step;
    total     = args.total;
    interval  = total // step;
    results   = dict();

    for i in range(args.fold):
        rated, popular = get_history(os.path.join(args.data, 'f%dtr.txt'%i));
        for feature in features:
            umat = get_mat(os.path.join(args.model, '%s%d/final-U.dat'%(feature, i)), uids);
            vmat = get_mat(os.path.join(args.model, '%s%d/final-V.dat'%(feature, i)), vids);
            bmat = None;
            if os.path.exists('%s%d/final-B.dat'%(feature, i)):
                bmat = get_mat(os.path.join(args.model, '%s%d/final-B.dat'%(feature, i)), vids)
            for scenario in scenarios:
                teids = get_ids(os.path.join(args.data, 'f%dte.%s.idl'%(i, scenario)));
                teivt = get_ivt(os.path.join(args.data, 'f%dte.%s.idl'%(i, scenario)));
                temat = np.zeros((len(teids), vmat.shape[1]), dtype=np.float32);
                for vid in teids:
                    temat[teids[vid],:] = vmat[vids[vid],:];
                scores = np.dot(umat, temat.T);
                if bmat is not None:
                    scores += bmat.reshape((1,-1));
                rlist  = np.argsort(scores, axis=1);
                tresults = [0.0]*interval;
                tcount = 0;
                for line in open(os.path.join(args.data, 'f%dte.%s.txt'%(i, scenario))):
                    terms = line.strip().split(',');
                    uid   = terms[0];
                    likes = set();
                    idx   = 0;
                    for k in range(1, len(terms)):
                        vid  = terms[k].split(':')[0];
                        like = int(terms[k].split(':')[1]);
                        if like == 1:
                            likes.add(teids[vid]);
                    if len(likes) != 0:
                        hits = [0] * interval;
                        for t in range(len(teids)):
                            liid = rlist[uids[uid], len(teids)-t-1];
                            if teivt[liid] not in rated[uid]:
                                if liid in likes:
                                    j = idx // step;
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
                if feature not in results[scenario]:
                    results[scenario][feature] = [0.0]*interval;
                for k in range(interval):
                    results[scenario][feature][k] += tresults[k] / tcount;
    for scenario in scenarios:
        print (scenario);
        for feature in features:
            line = feature;
            for k in range(interval):
                line += ',%.6f'%(results[scenario][feature][k] / fold);
            print (line);

if __name__ == '__main__':
    main();
