import numpy as np
import os
import pickle
import scipy.sparse as ss
import tensorflow.compat.v1 as tf
import time
from .wmf import WMF
from utils import get_id_dict_from_file, export_embed_to_file

class DPM(WMF):
    def __init__(self, k, d, lu=0.01, lv=10, le=10e3, a=1, b=0.01):
        self.__sn = 'dpm'
        WMF.__init__(self, k, lu, lv, a, b)
        self.d  = d
        self.le = le

    def load_content_data(self, content_file, iid_file):
        print ('Load content data from %s'%(content_file))
        fiids  = get_id_dict_from_file(iid_file)
        self.F = np.zeros((self.n_items, self.d), dtype = np.float32)
        F      = pickle.load(open(content_file, 'rb'), encoding = 'latin1')
        if ss.issparse(F):
            F = F.toarray()
        for iid in self.iids:
            if iid in fiids:
                self.F[self.iids[iid], :]=F[fiids[iid], :]
        print('Loading finished!')

    def train(self, model_path, encoder, max_iter = 200):
        loss  = np.exp(50)
        Ik    = np.eye(self.k, dtype = np.float32)
        with tf.Graph().as_default():
            self.encoder = encoder(self.k, self.d)
            sess = tf.Session(config = self.tf_config)
            sess.run(tf.global_variables_initializer())
            with sess.as_default():
                for it in range(max_iter):
                    t1     = time.time()
                    self.V = self.encoder.forward(sess, self.F)
                    loss_old = loss
                    loss     = 0
                    Vr = self.V[np.array(self.i_rated), :]
                    XX = np.dot(Vr.T, Vr)*self.b + Ik*self.lu
                    for i in self.usm:
                        if len(self.usm[i]) > 0:
                            Vi = self.V[np.array(self.usm[i]), :]
                            self.U[i,:] = np.linalg.solve(np.dot(Vi.T, Vi)*(self.a-self.b)+XX, np.sum(Vi, axis=0)*self.a)
                            loss += 0.5 * self.lu * np.sum(self.U[i,:]**2)
                    Ur = self.U[np.array(self.u_rated), :]
                    XX = np.dot(Ur.T, Ur)*self.b
                    for j in self.ism:
                        B  = XX
                        Fe = self.V[j,:].copy()
                        if len(self.ism[j]) > 0:
                            Uj = self.U[np.array(self.ism[j]), :]
                            B += np.dot(Uj.T, Uj)*(self.a-self.b)
                            self.V[j,:] = np.linalg.solve(B+Ik*self.lv, np.sum(Uj, axis=0)*self.a + Fe*self.lv)
                            loss += 0.5 * np.linalg.multi_dot((self.V[j, :], B, self.V[j, :]))
                            loss += 0.5 * len(self.ism[j])*self.a
                            loss -= np.sum(np.multiply(Uj, self.V[j, :]))*self.a
                        else:
                            self.V[j,:] = np.linalg.solve(B+Ik*self.lv, Fe*self.lv)
                        loss += 0.5 * self.lv * np.sum((self.V[j, :] - Fe)**2)
                    loss += self.encoder.backward(sess, self.F, self.V)
                    print ('Iter %3d, loss %.6f, time %.2fs'%(it, loss, time.time()-t1))
                if os.path.exists(os.path.dirname(model_path)):
                    print ('Saving model to path %s'%(model_path))
                    Fe = self.encoder.forward(sess, self.F)
                    for iidx in self.ism:
                        if iidx not in self.i_rated:
                            self.V[iidx, :] = Fe[iidx, :]
                    export_embed_to_file(os.path.join(model_path, 'final-U.dat'), self.U)
                    export_embed_to_file(os.path.join(model_path, 'final-V.dat'), self.V)
