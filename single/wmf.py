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
        ik   = np.eye(self.k, dtype=np.double)
        xx   = np.zeros((self.k, self.k), dtype=np.double)
        if model_path is not None and os.path.isdir(model_path):
            self.import_embeddings(model_path)
        for it in range(max_iter):
            t1 = time.time()
            loss_old, loss = loss, 0
            vr = self.fie[np.array(self.i_rated),:]
            np.dot(vr.T, vr*self.b, out=xx)
            xx += ik*self.lu
            for uid in self.usm:
                if len(self.usm[uid]) > 0:
                    v = self.fie[np.array(list(self.usm[uid].keys()))]
                    r = np.array(list(self.usm[uid].values()))
                    self.fue[uid] = np.linalg.solve(
                        xx+np.dot(v.T, v)*(self.a-self.b)
                        , np.sum(v*r.reshape(-1,1), axis=0)*self.a
                    )
                loss += 0.5 * self.lu * np.sum(self.fue[uid]**2)
            ur = self.fue[np.array(self.u_rated),:]
            np.dot(ur.T, ur*self.b, out=xx)
            for iid in self.ism:
                if len(self.ism[iid]) > 0:
                    u = self.fue[np.array(list(self.ism[iid].keys()))]
                    r = np.array(list(self.ism[iid].values()))
                    A  = xx.copy()
                    A += np.dot(u.T, u)*(self.a-self.b)
                    self.fie[iid] = np.linalg.solve(
                        A+ik*self.lv
                        , np.sum(u*r.reshape(-1,1), axis=0)*self.a
                    )
                    loss += 0.5 * np.sum(r**2) * self.a
                    loss += 0.5 * np.linalg.multi_dot((self.fie[iid], A, self.fie[iid]))
                    loss -= np.sum(np.dot(u, self.fie[iid])*r)*self.a
                loss += 0.5 * self.lv * np.sum(self.fie[iid]**2)
            cond = np.abs(loss_old - loss) / loss_old
            tprint('Iter %3d, loss %.6f, converge %.6f, time %.2fs'%(it, loss, cond, time.time()-t1))
            if cond < tol:
                break

    def export_model(self, model_path: str) -> None:
        return

    def import_model(self, model_path: str) -> None:
        return
