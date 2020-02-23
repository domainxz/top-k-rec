"""
    Collaborative Embedding Regression (CER)
    Sampling Method : uniform item sampling per user
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
    Reference       : "Personalized Video Recommendation Using Rich Contents from Videos", Xingzhong Du, et al.
"""

import numpy as np
import os
import time
from .wmf import WMF
from utils import tprint, get_embed_from_file, export_embed_to_file


class CER(WMF):
    def __init__(self, k: int, d: int, lu: float = 0.01, lv: float = 10, le: float = 10e3, a: float = 1, b: float = 0.01) -> None:
        super().__init__(k, lu, lv, a, b)
        self.__sn = 'cer'
        self.d = d
        self.le = le
        self.E = None

    def train(self, max_iter: int = 200, tol: float = 1e-4, model_path: str = None) -> None:
        loss = np.exp(50)
        Ik = np.eye(self.k, dtype=np.float32)
        FF = self.lv * np.dot(self.feat.T, self.feat) + self.le * np.eye(self.feat.shape[1])
        if model_path is not None and os.path.isdir(model_path):
            self.import_embeddings(model_path)
        if self.E is None:
            self.E = np.random.randn(self.feat.shape[1], self.k).astype(np.float32)
        for it in range(max_iter):
            t1 = time.time()
            Fe = np.dot(self.feat, self.E)
            loss_old = loss
            loss = 0
            Vr = self.fie[np.array(self.i_rated), :]
            XX = np.dot(Vr.T, Vr) * self.b + Ik * self.lu
            for i in self.usm:
                if len(self.usm[i]) > 0:
                    Vi = self.fie[np.array(self.usm[i]), :]
                    self.fue[i, :] = np.linalg.solve(
                            np.dot(Vi.T, Vi) * (self.a - self.b) + XX,
                            np.sum(Vi, axis=0) * self.a
                        )
                loss += 0.5 * self.lu * np.sum(self.fue[i, :] ** 2)
            Ur = self.fue[np.array(self.u_rated), :]
            XX = np.dot(Ur.T, Ur) * self.b
            for j in self.ism:
                B = XX.copy()
                if len(self.ism[j]) > 0:
                    Uj = self.fue[np.array(self.ism[j]), :]
                    B += np.dot(Uj.T, Uj) * (self.a - self.b)
                    self.fie[j, :] = np.linalg.solve(
                            B + Ik * self.lv, 
                            np.sum(Uj, axis=0) * self.a + Fe[j, :] * self.lv
                        )
                    loss += 0.5 * np.linalg.multi_dot((self.fie[j, :], B, self.fie[j, :]))
                    loss += 0.5 * len(self.ism[j]) * self.a
                    loss -= np.sum(np.multiply(Uj, self.fie[j, :])) * self.a
                else:
                    self.fie[j, :] = np.linalg.solve(B + Ik * self.lv, Fe[j, :] * self.lv)
                loss += 0.5 * self.lv * np.sum((self.fie[j, :] - Fe[j, :]) ** 2)
            self.E = np.linalg.solve(FF, self.lv * np.dot(self.feat.T, self.fie))
            loss += 0.5 * self.le * np.sum(self.E ** 2)
            cond = np.abs(loss_old - loss) / loss_old
            tprint('Iter %3d, loss %.6f, time %.2fs' % (it, loss, time.time() - t1))
            if cond < tol:
                break
        Fe = np.dot(self.feat, self.E)
        for iidx in self.ism:
            if iidx not in self.i_rated:
                self.fie[iidx, :] = Fe[iidx, :]

    def import_model(self, model_path: str) -> None:
        file_path = os.path.join(model_path, 'final-E.dat')
        if os.path.exists(file_path):
            tprint('Loading content projection matrix from %s' % file_path)
            self.E = get_embed_from_file(file_path)

    def export_model(self, model_path: str) -> None:
        if os.path.exists(model_path):
            if hasattr(self, 'E'):
                tprint('Saving content projection matrix to %s' % os.path.join(model_path, 'final-E.dat'))
                export_embed_to_file(os.path.join(model_path, 'final-E.dat'), self.E)
