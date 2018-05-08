"""
    Collaborative Regression (CR) through multiple layer perception (MLP)
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
    Reference       : "Deep content-based music recommendation", Aaron van den Oord, Sander Dieleman, Benjamin Schrauwen
"""

import argparse
import cPickle as pickle
import numpy as np
import numpy.linalg as la
import os
import scipy.sparse as ss
import theano
import theano.tensor as T
from predict_by_mlp import MLP

def getIdMap(idpath):
    idmap = dict();
    for line in open(idpath):
        vid = int(line.strip());
        idmap[vid] = len(idmap);
    return idmap;

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
    parser = argparse.ArgumentParser(description="Run dpm on a specific feature");
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

    cmd = '../cr/cr --directory %s --user ../complete/f%dtr-users.mfp --item ../complete/f%dtr-items.mfp --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --save_lag 1700 --random_seed 123 --theta_init %s/final.gamma >> %s/%s'

    start       = 0;
    end         = args.fold;
    fold        = args.fold;
    iterations  = args.iterations;
    directory   = os.path.join(args.work_dir, '%s%%d'%(args.fname));
    max_iter    = 1;
    num_factors = args.dimension;
    lv          = args.lambda_v;
    lu          = args.lambda_u;

    vmap = getIdMap('../complete/vid');
    mfcc = pickle.load(open(args.fpath, 'rb'));
    if ss.issparse(mfcc):
        mfcc = mfcc.toarray().astype(np.float32);
    batch_size = 128;
    n_hidden   = 2000;
    learning_rate = 0.0001
    L1_reg        = 0.00
    L2_reg        = 0.0001
    rng           = np.random.RandomState(2016);
    n_train_batches = int(np.ceil(mfcc.shape[0] * 1.0 / batch_size));
    for i in range(start, end):
        wdir = directory%(i);
        if not os.path.exists(wdir):
            os.mkdir(wdir);
        tevids  = getIdMap('../complete/f%dte.%s.idl'%(i, 'om'));
        train_x = theano.shared(mfcc, 'train_x');
        train_y = theano.shared(readVl('%s/init-V.dat'%wdir), 'train_y');
        left  = T.lscalar()
        right = T.lscalar()
        x = T.matrix('x');
        y = T.matrix('y');
        model = MLP(
            rng      = rng,
            input    = x,
            n_in     = mfcc.shape[1],
            n_hidden = n_hidden,
            n_out    = num_factors
        )
        cost = (
            model.errors(y)
            + L1_reg * model.L1
            + L2_reg * model.L2_sqr
        );
        gparams = [T.grad(cost, param) for param in model.params];
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(model.params, gparams)
        ]
        train_model = theano.function(
            inputs=[left, right],
            outputs=cost,
            updates=updates,
            givens={
                x: train_x[left:right],
                y: train_y[left:right]
            }
        )
        test_model = theano.function(
            inputs=[left, right],
            outputs=model.output,
            givens={
                x: train_x[left:right],
            }
        )
        for j in range(iterations):
            mlp_loss = 0;
            if j != 0:
                train_y = theano.shared(readVl('%s/final-V.dat'%wdir), 'train_y');
            for minibatch_index in range(n_train_batches):
                mlp_loss += train_model(minibatch_index * batch_size, min((minibatch_index + 1) * batch_size, mfcc.shape[0]));
            Vl = np.vstack([test_model(k * batch_size, min((k+1) * batch_size, mfcc.shape[0])) for k in range(n_train_batches)]); 
            if j == 0:
                writeVl(Vl, '%s/final-V.dat'%wdir);
            writeVl(Vl, '%s/final.gamma'%wdir);
            os.system(cmd%(wdir, i, i, max_iter, num_factors, lv, lu, wdir, wdir, 'state.log'));
            ctr_loss = float(open('%s/final-likelihood.dat'%wdir).readlines()[0].strip());
            total_loss = mlp_loss - ctr_loss;
            print 'Iter %4d, clr likelihood:%f, total loss:%f'%(j+1, ctr_loss, total_loss);
        Vim = readVl('%s/final-V.dat'%wdir);
        Vom = np.vstack([test_model(k * batch_size, min((k+1) * batch_size, mfcc.shape[0])) for k in range(n_train_batches)]);
        for vid in vmap:
            if vid in tevids:
                Vim[vmap[vid],:] = Vom[vmap[vid],:];
    writeVl(Vim, '%s/final-V.dat'%wdir);
    print 'Training on fold %d with %s finished!'%(i, args.fname);

if __name__ == '__main__':
    main();
