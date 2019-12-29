from .encoder import ENCODER
import numpy as np
import tensorflow.compat.v1 as tf
import time
from utils import tprint
from .wmf import WMF


class DPM(WMF):
    def __init__(self, k: int, d: int, lu: float = 0.01, lv: float = 10, le: float = 10e3, a: float = 1, b: float = 0.01) -> None:
        self.__sn = 'dpm'
        WMF.__init__(self, k, lu, lv, a, b)
        self.d = d
        self.le = le

    def train(self, encoder: ENCODER, max_iter: int = 200) -> None:
        loss = np.exp(50)
        Ik = np.eye(self.k, dtype=np.float32)
        with tf.Graph().as_default():
            self.encoder = encoder(self.k, self.d)
            sess = tf.Session(config=self.tf_config)
            sess.run(tf.global_variables_initializer())
            with sess.as_default():
                for it in range(max_iter):
                    t1 = time.time()
                    self.fie = self.encoder.out(sess, self.feat)
                    loss_old = loss
                    loss = 0
                    Vr = self.fie[np.array(self.i_rated), :]
                    XX = np.dot(Vr.T, Vr) * self.b + Ik * self.lu
                    for i in self.usm:
                        if len(self.usm[i]) > 0:
                            Vi = self.fie[np.array(self.usm[i]), :]
                            self.fue[i, :] = np.linalg.solve(np.dot(Vi.T, Vi) * (self.a - self.b) + XX,
                                                           np.sum(Vi, axis=0) * self.a)
                            loss += 0.5 * self.lu * np.sum(self.fue[i, :] ** 2)
                    Ur = self.fue[np.array(self.u_rated), :]
                    XX = np.dot(Ur.T, Ur) * self.b
                    for j in self.ism:
                        B = XX
                        Fe = self.fie[j, :].copy()
                        if len(self.ism[j]) > 0:
                            Uj = self.fue[np.array(self.ism[j]), :]
                            B += np.dot(Uj.T, Uj) * (self.a - self.b)
                            self.fie[j, :] = np.linalg.solve(B + Ik * self.lv, np.sum(Uj, axis=0) * self.a + Fe * self.lv)
                            loss += 0.5 * np.linalg.multi_dot((self.fie[j, :], B, self.fie[j, :]))
                            loss += 0.5 * len(self.ism[j]) * self.a
                            loss -= np.sum(np.multiply(Uj, self.fie[j, :])) * self.a
                        else:
                            self.fie[j, :] = np.linalg.solve(B + Ik * self.lv, Fe * self.lv)
                        loss += 0.5 * self.lv * np.sum((self.fie[j, :] - Fe) ** 2)
                    loss += self.encoder.fit(sess, self.feat, self.fie)
                    tprint('Iter %3d, loss %.6f, time %.2fs' % (it, loss, time.time() - t1))
        Fe = self.encoder.out(sess, self.feat)
        for iidx in self.ism:
            if iidx not in self.i_rated:
                self.fie[iidx, :] = Fe[iidx, :]
