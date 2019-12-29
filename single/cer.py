"""
    Collaborative Embedding Regression (CER)
    Sampling Method : uniform item sampling per user
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
    Reference       : "Personalized Video Recommendation Using Rich Contents from Videos", Xingzhong Du, et al.
"""

import numpy as np
import time
from .wmf import WMF
from utils import tprint


class CER(WMF):
    def __init__(self, k: int, d: int, lu: float = 0.01, lv: float = 10, le: float = 10e3, a: float = 1, b: float = 0.01):
        self.__sn = 'cer'
        WMF.__init__(self, k, lu, lv, a, b)
        self.d = d
        self.le = le

    def train(self, max_iter: int = 200):
        loss = np.exp(50)
        Ik = np.eye(self.k, dtype=np.float32)
        FF = self.lv * np.dot(self.feat.T, self.feat) + self.le * np.eye(self.feat.shape[1])
        self.E = np.random.randn(self.feat.shape[1], self.k).astype(np.float32)
        for it in range(max_iter):
            t1 = time.time()
            self.fie = np.dot(self.feat, self.E)
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
            self.E = np.linalg.solve(FF, self.lv * np.dot(self.feat.T, self.fie))
            loss += 0.5 * self.le * np.sum(self.E ** 2)
            tprint('Iter %3d, loss %.6f, time %.2fs' % (it, loss, time.time() - t1))
        Fe = np.dot(self.feat, self.E)
        for iidx in self.ism:
            if iidx not in self.i_rated:
                self.fie[iidx, :] = Fe[iidx, :]
