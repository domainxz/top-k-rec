from abc import ABC, abstractmethod
import numpy as np
from .rec import REC
import os
import tensorflow.compat.v1 as tf
import time
from utils import get_id_dict_from_file,  tprint


class WMF(REC):
    def __init__(self, k: int, lu: float = 0.01, lv: float = 0.01, a: float = 1, b: float = 0.01) -> None:
        self.__sn = 'wmf'
        self.k = k
        self.lu = lu
        self.lv = lv
        self.a  = a
        self.b  = b
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth=True
        self.uids = None
        self.n_users = None
        self.usm = None
        self.iids = None
        self.n_items = None
        self.ism = None
        self.n_ratings = None
        self.u_rated = None
        self.i_rated = None
        self.fue = None
        self.fie = None

    def load_training_data(self, uid_file: str, iid_file: str, tr_file: str) -> None:
        self.uids = get_id_dict_from_file(uid_file)
        self.n_users = len(self.uids)
        self.usm = dict()
        for uid in self.uids.values():
            self.usm[uid] = list()
        self.iids = get_id_dict_from_file(iid_file)
        self.n_items = len(self.iids)
        self.n_ratings = self.n_users * self.n_items
        self.ism = dict()
        for iid in self.iids.values():
            self.ism[iid] = list()
        for line in open(tr_file, 'r'):
            terms = line.strip().split(',')
            uid   = terms[0]
            for j in range(1, len(terms)):
                iid  = terms[j].split(':')[0]
                like = terms[j].split(':')[1]
                if like == '1':
                    self.usm[self.uids[uid]].append(self.iids[iid])
                    self.ism[self.iids[iid]].append(self.uids[uid])
        self.u_rated = [uidx for uidx in self.usm if len(self.usm[uidx]) > 0]
        self.i_rated = [iidx for iidx in self.ism if len(self.ism[iidx]) > 0]
        self.fue = np.random.rand(self.n_users, self.k).astype(np.float32)
        self.fie = np.random.rand(self.n_items, self.k).astype(np.float32)

    def build_graph(self) -> None:
        tprint('%s does not require build_graph method!' % self.__sn)

    def train(self, max_iter: int = 200, tol: float = 1e-4, model_path: str = None) -> None:
        loss = np.exp(50)
        Ik   = np.eye(self.k, dtype=np.float32)
        if model_path is not None and os.path.isdir(model_path):
            self.import_embeddings(model_path)
        for it in range(max_iter):
            t1 = time.time()
            loss_old = loss
            loss     = 0
            Vr = self.fie[np.array(self.i_rated), :]
            XX = np.dot(Vr.T, Vr)*self.b + Ik*self.lu
            for i in self.usm:
                if len(self.usm[i]) > 0:
                    Vi = self.fie[np.array(self.usm[i]), :]
                    self.fue[i, :] = np.linalg.solve(np.dot(Vi.T, Vi)*(self.a-self.b)+XX, np.sum(Vi, axis=0)*self.a)
                loss += 0.5 * self.lu * np.sum(self.fue[i,:]**2)
            Ur = self.fue[np.array(self.u_rated), :]
            XX = np.dot(Ur.T, Ur)*self.b
            for j in self.ism:
                if len(self.ism[j]) > 0:
                    Uj = self.fue[np.array(self.ism[j]), :]
                    B  = np.dot(Uj.T, Uj)*(self.a-self.b) + XX 
                    self.fie[j, :] = np.linalg.solve(B+Ik*self.lv, np.sum(Uj, axis = 0) * self.a)
                    loss += 0.5 * len(self.ism[j])*self.a
                    loss += 0.5 * np.linalg.multi_dot((self.fie[j, :], B, self.fie[j, :]))
                    loss -= np.sum(np.multiply(Uj, self.fie[j, :]))*self.a
                loss += 0.5 * self.lv * np.sum(self.fie[j, :]**2)
            cond = np.abs(loss_old - loss) / loss_old
            tprint('Iter %3d, loss %.6f, converge %.6f, time %.2fs'%(it, loss, cond, time.time()-t1))
            if cond < tol:
                break

    def export_model(self, model_path: str) -> None:
        return

    def import_model(self, model_path: str) -> None:
        return
