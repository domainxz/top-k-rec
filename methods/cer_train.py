"""
    Collaborative Regression (CR) through linear embedding
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
"""

import argparse
import cPickle as pickle
import numpy as np
import numpy.linalg as la
import scipy.sparse as ss
import os

def readVl(vpath):
    V = None;
    lines = open(vpath).readlines();
    for i in range(len(lines)):
        terms = lines[i].strip().split(' ');
        if V is None:
            V = np.zeros((len(lines), len(terms)), dtype = np.float32);
        for j in range(len(terms)):
            V[i,j] = float(terms[j]);
    return V;

def writeVl(V, vpath):
    fid = open(vpath, 'w');
    row, col = V.shape;
    for i in range(row):
        for j in range(col):
            fid.write('%f '%V[i,j]);
        fid.write('\n');
    fid.close();

def main():
    parser = argparse.ArgumentParser(description="Run cer on a specific feature");
    parser.add_argument('-fp', '--fpath', required=True, help="The feature path");
    parser.add_argument('-fn', '--fname', required=True, help="The feature name");
    parser.add_argument('-wd', '--work_dir', required=True, help="The working directory");
    parser.add_argument('-f', '--fold', type=int, default=5, help="The fold number");
    parser.add_argument('-d', '--dimension', type=int, default=50, help="The latent factor dimension");
    parser.add_argument('-i', '--iterations', type=int, default=200, help="The iteration number");
    parser.add_argument('-lv', '--lambda_v', type=float, default=10, help="Regularization parameter on V");
    parser.add_argument('-lu', '--lambda_u', type=float, default=0.1, help="Regularization parameter on U");
    parser.add_argument('-le', '--lambda_e', type=float, default=1.0e3, help="Regularization parameter on E");
    args = parser.parse_args();

    cmd = '../cr/cr --directory %s --user ../complete/f%dtr-users.mfp --item ../complete/f%dtr-items.mfp --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --save_lag 1700 --random_seed 123 --theta_init %s/final.gamma >> %s/%s';
    fold        = args.fold;
    iterations  = args.iterations;
    directory   = os.path.join(args.work_dir, '%s%%d'%(args.fname));
    max_iter    = 1;
    num_factors = args.dimension;
    lv          = args.lambda_v;
    lu          = args.lambda_u;
    le          = args.lambda_e;
    vmap        = dict();
    for line in open('../complete/vid'):
        vid = int(line.strip());
        vmap[vid] = len(vmap);

    V = pickle.load(open(args.fpath));
    if ss.issparse(V):
        V = V.toarray().astype(np.float32);
    I = np.eye(V.shape[1], dtype=np.float32);
    invVV = la.inv(lv * np.dot(V.T, V) + le * I);
    for i in range(fold):
        E    = np.random.randn(V.shape[1], num_factors).astype(np.float32);
        wdir = directory%(i);
        if not os.path.exists(wdir):
            os.mkdir(wdir);
        for j in range(iterations):
            Vl = np.dot(V, E);
            if j == 0:
                writeVl(Vl, '%s/final-V.dat'%wdir);
            writeVl(Vl, '%s/final.gamma'%wdir);
            os.system(cmd%(wdir, i, i, max_iter, num_factors, lv, lu, wdir, wdir, 'state.log'));
            Vl = readVl('%s/final-V.dat'%wdir);
            E  = np.dot(invVV, lv * np.dot(V.T, Vl));
            emb_loss = le * (E ** 2).sum() + lv * ((Vl - np.dot(V, E)) ** 2).sum();
            ctr_loss = float(open('%s/final-likelihood.dat'%wdir).readlines()[0].strip());
            total_loss = emb_loss - ctr_loss;
            print 'Iter %4d, ctr likelihood:%f, total loss:%f'%(j+1, ctr_loss, total_loss);
        tevids = set();
        for line in open('../complete/f%dte.%s.idl'%(i, 'om')):
            vid = int(line.strip());
            tevids.add(vid);
        Vim = readVl('%s/final-V.dat'%wdir);
        Vom = np.dot(V, E);
        for vid in vmap:
            if vid in tevids:
                Vim[vmap[vid],:] = Vom[vmap[vid],:];
        writeVl(Vim, '%s/final-V.dat'%wdir);
        print 'Training on fold %d with %s finished!'%(i, args.fname);

if __name__ == '__main__':
    main();
