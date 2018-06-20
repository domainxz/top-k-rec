from abc import ABC, abstractmethod
import numpy as np
import os
import tensorflow as tf
import time
from utils import get_id_dict_from_file, export_embed_to_file

class WMF(ABC):
    def __init__(self, k, lu=0.01, lv=0.01, a=1, b=0.01):
        self.__sn = 'wmf';
        self.k = k;
        self.lu = lu;
        self.lv = lv;
        self.a  = a;
        self.b  = b;
        self.tf_config = tf.ConfigProto();
        self.tf_config.gpu_options.allow_growth=True;

    def load_train_data(self, upath, ipath, trpath):
        self.uids = get_id_dict_from_file(upath);
        self.n_users = len(self.uids);
        self.usm = dict();
        for uid in self.uids.values():
            self.usm[uid] = list();
        self.iids = get_id_dict_from_file(ipath);
        self.n_items = len(self.iids);
        self.n_ratings = self.n_users * self.n_items;
        self.ism = dict();
        for iid in self.iids.values():
            self.ism[iid] = list();
        for line in open(trpath, 'r'):
            terms = line.strip().split(',');
            uid   = terms[0];
            for j in range(1, len(terms)):
                iid  = terms[j].split(':')[0];
                like = terms[j].split(':')[1];
                if like == '1':
                    self.usm[self.uids[uid]].append(self.iids[iid]);
                    self.ism[self.iids[iid]].append(self.uids[uid]);
        self.u_rated = [uidx for uidx in self.usm if len(self.usm[uidx]) > 0];
        self.i_rated = [iidx for iidx in self.ism if len(self.ism[iidx]) > 0];
        self.U = np.random.rand(self.n_users, self.k).astype(np.float32);
        self.V = np.random.rand(self.n_items, self.k).astype(np.float32);

    def train(self, model_path, max_iter=200, tol=1e-4):
        loss = np.exp(50);
        Ik   = np.eye(self.k, dtype=np.float32);
        for it in range(max_iter):
            t1 = time.time();
            loss_old = loss;
            loss     = 0;
            Vr = self.V[np.array(self.i_rated), :];
            XX = np.dot(Vr.T, Vr)*self.b + Ik*self.lu;
            for i in self.usm:
                if len(self.usm[i]) > 0:
                    Vi = self.V[np.array(self.usm[i]), :];
                    self.U[i,:] = np.linalg.solve(np.dot(Vi.T, Vi)*(self.a-self.b)+XX, np.sum(Vi, axis=0)*self.a);
                    loss += 0.5 * self.lu * np.sum(self.U[i,:]**2);
            Ur = self.U[np.array(self.u_rated), :];
            XX = np.dot(Ur.T, Ur)*self.b
            for j in self.ism:
                if len(self.ism[j]) > 0:
                    Uj = self.U[np.array(self.ism[j]), :];
                    B  = np.dot(Uj.T, Uj)*(self.a-self.b) + XX; 
                    self.V[j,:] = np.linalg.solve(B+Ik*self.lv, np.sum(Uj, axis=0)*self.a);
                    loss += 0.5 * len(self.ism[j])*self.a;
                    loss += 0.5 * np.linalg.multi_dot((self.V[j,:], B, self.V[j,:]));
                    loss -= np.sum(np.multiply(Uj, self.V[j,:]))*self.a;
                    loss += 0.5 * self.lv * np.sum(self.V[j,:]**2);
            cond = np.abs(loss_old - loss) / loss_old;
            print ('Iter %3d, loss %.6f, converge %.6f, time %.2fs'%(it, loss, cond, time.time()-t1));
            if cond < tol:
                break;
        if os.path.exists(os.path.dirname(model_path)):
            print ('Saving model to path %s'%(model_path))
            export_embed_to_file(os.path.join(model_path, 'final-U.dat'), self.U);
            export_embed_to_file(os.path.join(model_path, 'final-V.dat'), self.V);
