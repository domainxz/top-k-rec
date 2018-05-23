import numpy as np
import os

def get_id_dict_from_file(file_path):
    ids = dict();
    for line in open(file_path, 'r'):
        tid = line.strip();
        ids[tid] = len(ids);
    return ids;

def get_iv_dict_from_file(file_path):
    ivt = dict();
    for line in open(file_path):
        tid = line.strip();
        ivt[len(ivt)] = tid;
    return ivt;

def get_embed_from_file(file_path, ids):
    embed = None;
    lines = open(file_path).readlines();
    for tid in ids:
        terms = lines[ids[tid]].strip().split(' ');
        if embed is None:
            embed = np.zeros((len(ids), len(terms)), dtype=np.float32);
        for k in range(len(terms)):
            embed[ids[tid], k] = np.float32(terms[k]);
    return mat;

def get_train_data_from_file(file_path):
    data = list();
    tr_uids = dict();
    tr_iids = dict();
    for line in open(file_path, 'r'):
        terms = line.strip().split(',');
        uid   = terms[0];
        if len(terms) > 1:
            tr_uids[uid] = len(tr_uids);
            for j in range(1, len(terms)):
                iid  = terms[j].split(':')[0];
                like = terms[j].split(':')[1];
                if iid not in tr_iids:
                    tr_iids[iid] = len(tr_iids);
                if like == '1':
                    data.append((uid, iid));
    return data, tr_uids, tr_iids;

def get_history_from_file(file_path):
    browsed = dict();
    counter = dict();
    for line in open(file_path):
        terms = line.strip().split(',');
        uid   = terms[0];
        browsed[uid] = set();
        for k in range(1, len(terms)):
            iid  = terms[k].split(':')[0];
            like = terms[k].split(':')[1];
            browsed[uid].add(iid);
            if like == '1':
                if iid not in counter:
                    counter[iid] = 0;
                counter[iid] += 1;
    return browsed, counter

def get_score(U, V, iids, sub_iids):
    subV = np.zeros((len(sub_iids), V.shape[1]), dtype=np.float32);
    for iid in iids:
        if iid in sub_iids:
            subV[sub_iids[iid],:] = V[iids[iid],:];
    score = np.dot(U, subV.T);
    return score;

def evaluate(score, rated, likes, uids, te_iids, te_ivt, step, total, interval):
    count = 0;
    hits  = [0.0] * interval;
    trrs  = [0.0] * interval;
    ranks = np.argsort(score, axis=1);
    for uid in likes:
        idx = 0;
        like = likes[uid];
        if len(like) != 0:
            hit = [0.0] * interval;
            rrs = [0.0] * interval;
            for t in range(len(te_iids)):
                riid = te_ivt[ranks[uids[uid], len(te_iids)-1-t]];
                if riid not in rated[uid]:
                    if riid in like:
                        j = t // step;
                        for k in range(j, interval):
                            hit[k] += 1;
                            rrs[k] += 1.0 / (t+1);
                    idx += 1;
                if idx == total:
                    break;
            for k in range(interval):
                hits[k] += hit[k];
                trrs[k] += rrs[k];
            count += len(like);
    return hits, trrs, count
