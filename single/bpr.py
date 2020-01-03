"""
    Matrix Factorization (MF) based on Bayesian Personalized Ranking (BPR)
    Sampling Method : uniform item sampling per user
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
    Reference       : "BPR : Bayesian Personalized Ranking from Implicit Feedback", Steven Rendle, etc.
"""

from collections import defaultdict
from .rec import REC
import numpy as np
import os
import sys
import tensorflow.compat.v1 as tf
import time
from utils import tprint, get_id_dict_from_file, get_data_from_file


class BPR(REC):
    def __init__(self, k: int, lambda_u: float = 2.5e-3, lambda_i: float = 2.5e-3, lambda_j: float = 2.5e-4, lambda_b: float = 0, lr: float = 1.0e-4, mode: str = 'l2') -> None:
        self.k = k
        self.lu = lambda_u
        self.li = lambda_i
        self.lj = lambda_j
        self.lb = lambda_b
        self.lr = lr
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.uids = None
        self.iids = None
        self.data = None
        self.epoch_sample_limit = None
        self.n_users = None
        self.n_items = None
        self.tr_data = None
        self.tr_users = None
        self.__ue = None
        self.__ie = None
        self.__ib = None
        self.__sess = None
        self.__saver = None
        self.mode = mode
        self.pred = None
        self.obj = None
        self.solver = None
        self.fue = None
        self.fie = None
        self.fib = None


    def load_training_data(self, uid_file: str, iid_file: str, tr_file: str, data_copy: bool = False) -> None:
        tprint('Load training data from %s' % (tr_file))
        self.uids = get_id_dict_from_file(uid_file)
        self.iids = get_id_dict_from_file(iid_file)
        self.data = get_data_from_file(tr_file, self.uids, self.iids)
        self.epoch_sample_limit = len(self.data)
        assert isinstance(self.uids, dict)
        assert isinstance(self.iids, dict)
        self.n_users = len(self.uids)
        assert self.n_users > 0
        self.n_items = len(self.iids)
        assert self.n_items > 0
        self.tr_data = self._data_to_training_dict(self.data, self.uids, self.iids)
        assert isinstance(self.tr_data, dict)
        self.tr_users = list(self.tr_data.keys())

        if not data_copy:
            del self.data
        tprint('Loading finished!')

    def build_graph(self) -> 'List[tf.placeholder[tf.int32]]':
        with tf.variable_scope('bpr', reuse=tf.AUTO_REUSE):
            u = tf.placeholder(tf.int32, [None])
            i = tf.placeholder(tf.int32, [None])
            j = tf.placeholder(tf.int32, [None])

            self.__ue = tf.get_variable(name="user_embed", shape=[self.n_users, self.k], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
            self.__ie = tf.get_variable(name="item_embed", shape=[self.n_items, self.k], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
            self.__ib = tf.get_variable(name="item_bias",  shape=[self.n_items],         dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            
        ueb = tf.nn.embedding_lookup(self.__ue, u)
        ieb = tf.nn.embedding_lookup(self.__ie, i)
        jeb = tf.nn.embedding_lookup(self.__ie, j)
        ib  = tf.nn.embedding_lookup(self.__ib, i)
        jb  = tf.nn.embedding_lookup(self.__ib, j)

        x_ui  = tf.reduce_sum(tf.multiply(ueb, ieb), 1)
        x_uj  = tf.reduce_sum(tf.multiply(ueb, jeb), 1)
        x_uij = ib - jb + x_ui - x_uj
        with tf.name_scope('output'):
            self.pred = tf.matmul(ueb, tf.transpose(ieb)) + ib
            if self.mode == 'l2':
                self.obj = tf.reduce_sum(tf.log(1+tf.exp(-x_uij)))+\
                           0.5 * tf.reduce_sum(ueb**2*self.lu+ieb**2*self.li+jeb**2*self.lj)+\
                           0.5 * tf.reduce_sum(ib**2+jb**2)*self.lb
            else:
                self.obj = tf.reduce_sum(tf.log(1+tf.exp(-x_uij)))+\
                           tf.reduce_sum(tf.abs(ueb)*self.lu+tf.abs(ieb)*self.li+tf.abs(jeb)*self.lj)+\
                           tf.reduce_sum(tf.abs(ib)+tf.abs(jb))*self.lb
        self.solver = tf.train.RMSPropOptimizer(self.lr).minimize(self.obj)
        return u, i, j

    def train(self, sampling: str = 'user uniform', epochs: int = 5, batch_size: int = 256, epoch_sample_limit: int = None, model_path: str = None):
        assert isinstance(sampling, str)
        assert isinstance(epochs, int)
        assert isinstance(batch_size, int)
        with tf.Graph().as_default():
            u, i, j = self.build_graph()
            self.__saver = tf.train.Saver()
            if epoch_sample_limit is not None:
                assert isinstance(epoch_sample_limit, int)
                self.epoch_sample_limit = epoch_sample_limit
            batch_limit = self.epoch_sample_limit//batch_size + 1
            self.__sess = tf.Session(config=self.tf_config)
            sampler = None
            if sampling == 'user uniform':
                sampler = self._uniform_user_sampling
            with self.__sess.as_default():
                self.__sess.run(tf.global_variables_initializer())
                if model_path is not None:
                    assert isinstance(model_path, str)
                    tprint("Initialize weights with the previous trained model")
                    self.import_embeddings(model_path)
                tprint('Training parameters: lu=%.6f, li=%.6f, lj=%.6f, lb=%.6f' % (self.lu, self.li, self.lj, self.lb))
                tprint('Learning rate is %.6f, regularization mode is %s' % (self.lr, self.mode))
                tprint('Training for %d epochs of %d batches using %s sampler' % (epochs, batch_limit, sampling))
                if self.fue is not None:
                    tprint('Initialize user embeddings')
                    self.__sess.run(tf.assign(self.__ue, self.fue))
                if self.fie is not None:
                    tprint('Initialize item embeddings')
                    self.__sess.run(tf.assign(self.__ie, self.fie))
                if self.fib is not None:
                    tprint('Initialize item biases')
                    self.__sess.run(tf.assign(self.__ib, self.fib.ravel()))
                for eid in range(epochs):
                    total_time = 0
                    bno = 1
                    for ub, ib, jb in sampler(batch_size):
                        t1 = time.time()
                        _, loss = self.__sess.run([self.solver, self.obj], feed_dict={u: ub, i: ib, j: jb})
                        t2 = time.time()-t1
                        sys.stderr.write('\rEpoch=%3d, batch=%6d, loss=%8.4f, time=%4.4fs' % (eid+1, bno, loss, t2))
                        total_time += t2
                        bno += 1
                        if bno == batch_limit:
                            break
                    sys.stderr.write(' ... total time collapse %8.4fs'%(total_time))
                    sys.stderr.flush()
                    print()
            self.fue = self.__sess.run(self.__ue)
            self.fie = self.__sess.run(self.__ie)
            self.fib = self.__sess.run(tf.reshape(self.__ib, (-1, 1)))
    
    def _uniform_user_sampling(self, batch_size: int) -> 'List[np.ndarray[np.int32]]':
        ib = np.zeros(batch_size, dtype=np.int32)
        jb = np.zeros(batch_size, dtype=np.int32)
        while True:
            ub = np.random.choice(self.tr_users, batch_size)
            for i in range(batch_size):
                ib[i] = np.random.choice(self.tr_data[ub[i]])
                jb[i] = np.random.choice(self.n_items)
                while jb[i] in self.tr_data[ub[i]]:
                    jb[i] = np.random.choice(self.n_items)
            yield ub, ib, jb
    
    def _data_to_training_dict(self, data: list, users: dict, items: dict) -> 'Defaultdict[list]':
        data_dict = defaultdict(list)
        for (user, item) in data:
            data_dict[users[user]].append(items[item])
        return data_dict

    def import_model(self, model_path: str) -> None:
        file_path = os.path.join(model_path, 'weights')
        if os.path.exists(model_path) and self.__sess is not None and self.__saver is not None:
            tprint('Restoring tensorflow graph from path %s' % (file_path))
            self.__saver.restore(self.__sess, file_path)

    def export_model(self, model_path: str) -> None:
        if os.path.exists(model_path) and self.__sess is not None and self.__saver is not None:
            file_path = os.path.join(model_path, 'weights')
            tprint('Saving tensorflow graph to path %s' % (file_path))
            self.__saver.save(self.__sess, file_path)
