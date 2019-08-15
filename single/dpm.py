import numpy as np
import os
import pickle
import scipy.sparse as ss
import tensorflow as tf
import time
from .wmf import WMF
from utils import get_id_dict_from_file, export_embed_to_file

class MLP:
    def __init__(self, k, d, lr = 1e-4, lbd = 1e-4, hidden_layers = [2000, 1000]):
        self._k = k
        self._d = d
        self._lr = lr
        self._lbd = lbd
        self._hidden_layers = hidden_layers
        self.x = tf.placeholder(tf.float32, [None, self._d])
        self.y = tf.placeholder(tf.float32, [None, self._k])
        with tf.variable_scope("content_mlp", reuse = tf.AUTO_REUSE):
            t = self.x
            for lid, num_of_units in enumerate(self._hidden_layers):
                t = tf.layers.dense(t, num_of_units, activation = tf.sigmoid, name = 'layer_%d'%lid)
            self.F = tf.layers.dense(t, self._k, name = 'layer_output')
            self.obj = 0.5 * tf.reduce_sum((self.y - self.F) ** 2)
            self.solver = tf.train.RMSPropOptimizer(self._lr).minimize(self.obj)

    def forward(self, sess, X, batch_size = 64):
        n_row, n_col = X.shape
        F = np.zeros((n_col, self._k), dtype=np.float32)
        for i in range(0, n_row, batch_size):
            actual_batch_size = min(batch_size, n_row-i)
            F[i: i+actual_batch_size, :] = sess.run(self.F, feed_dict={self.x: X[i: i+actual_batch_size]})
        return F

    def backward(self, sess, X, Y, batch_size = 64):
        n_row, n_col = X.shape
        ridxs = np.random.permutation(n_row)
        obj = 0
        for i in range(0, n_row, batch_size):
            actual_batch_size = min(batch_size, n_row-i)
            batch_obj, _ = sess.run([self.obj, self.solver], feed_dict={self.x: X[ridxs[i: i+actual_batch_size]], self.y: Y[ridxs[i: i+actual_batch_size]]})
            obj += batch_obj
        return obj

class DPM(WMF):
    def __init__(self, k, d, lu=0.01, lv=10, le=10e3, a=1, b=0.01):
        self.__sn = 'dpm';
        WMF.__init__(self, k, lu, lv, a, b);
        self.d  = d;
        self.le = le;

    def load_content_data(self, content_file, iid_file):
        print ('Load content data from %s'%(content_file));
        fiids  = get_id_dict_from_file(iid_file);
        self.F = np.zeros((self.n_items, self.d), dtype=np.float32);
        F      = pickle.load(open(content_file, 'rb'), encoding='latin1');
        if ss.issparse(F):
            F = F.toarray();
        for iid in self.iids:
            if iid in fiids:
                self.F[self.iids[iid],:]=F[fiids[iid],:];
        print('Loading finished!');

    def train(self, model_path, max_iter=200):
        loss  = np.exp(50);
        Ik    = np.eye(self.k, dtype=np.float32);
        with tf.Graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.mlp = MLP(self.k, self.d)
            sess = tf.Session(config = config)
            sess.run(tf.global_variables_initializer());
            with sess.as_default():
                for it in range(max_iter):
                    t1     = time.time()
                    self.V = self.mlp.forward(sess, self.F)
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
                            loss += 0.5 * np.linalg.multi_dot((self.V[j,:], B, self.V[j,:]))
                            loss += 0.5 * len(self.ism[j])*self.a
                            loss -= np.sum(np.multiply(Uj, self.V[j,:]))*self.a
                        else:
                            self.V[j,:] = np.linalg.solve(B+Ik*self.lv, Fe*self.lv)
                        loss += 0.5 * self.lv * np.sum((self.V[j,:] - Fe)**2)
                    loss += self.mlp.backward(sess, self.F, self.V)
                    print ('Iter %3d, loss %.6f, time %.2fs'%(it, loss, time.time()-t1))
                if os.path.exists(os.path.dirname(model_path)):
                    print ('Saving model to path %s'%(model_path))
                    Fe = self.mlp.forward(sess, self.F)
                    for iidx in self.ism:
                        if iidx not in self.i_rated:
                            self.V[iidx, :] = Fe[iidx, :]
                    export_embed_to_file(os.path.join(model_path, 'final-U.dat'), self.U)
                    export_embed_to_file(os.path.join(model_path, 'final-V.dat'), self.V)
