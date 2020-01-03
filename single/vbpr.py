"""
    Matrix Factorization (MF) based on Content-aware Bayesian Personalized Ranking (CBPR)
    Sampling Method : uniform item sampling per user
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
    Reference       : "VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback", Ruining He, Julian McAuley
"""

from .bpr import BPR
import os
import sys
import tensorflow.compat.v1 as tf
import time
from utils import tprint


class VBPR(BPR):
    def __init__(self, k: int, d: int, lambda_u: float = 2.5e-3, lambda_i: float = 2.5e-3, lambda_j: float = 2.5e-4, lambda_b: float = 0, lambda_e: float = 0, lr: float = 1.0e-4, mode: str = 'l2') -> None:
        super().__init__(k, lambda_u, lambda_i, lambda_j, lambda_b, lr, mode)
        self.d = d
        self.le = lambda_e
        self.__ure = None
        self.__uce = None
        self.__ire = None
        self.__irb = None
        self.__cem = None
        self.__icb = None
        
    def build_graph(self) -> 'List[tf.placeholder]':
        with tf.variable_scope('cbpr', reuse=tf.AUTO_REUSE):
            u  = tf.placeholder(tf.int32,   [None])
            i  = tf.placeholder(tf.int32,   [None])
            j  = tf.placeholder(tf.int32,   [None])
            ic = tf.placeholder(tf.float32, [None, self.d])
            jc = tf.placeholder(tf.float32, [None, self.d])

            self.__ure = tf.get_variable(name="user_rating_embed", shape=[self.n_users, self.k // 2], dtype=tf.float32,
                                         initializer=tf.random_normal_initializer(0, 0.01))
            self.__uce = tf.get_variable(name="user_content_embed", shape=[self.n_users, self.k // 2], dtype=tf.float32,
                                         initializer=tf.random_normal_initializer(0, 0.01))
            self.__ire = tf.get_variable(name="item_rating_embed", shape=[self.n_items, self.k // 2], dtype=tf.float32,
                                         initializer=tf.random_normal_initializer(0, 0.01))
            self.__irb = tf.get_variable(name="item_rating_bias", shape=[self.n_items, 1], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            self.__cem = tf.get_variable(name="content_embed_matrix", shape=[self.d, self.k // 2], dtype=tf.float32,
                                         initializer=tf.constant_initializer(2 / (self.d * self.k)))
            self.__icb = tf.get_variable(name="item_content_bias", shape=[self.d, 1], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))
            
        ureb = tf.nn.embedding_lookup(self.__ure, u)
        uceb = tf.nn.embedding_lookup(self.__uce, u)
        ireb = tf.nn.embedding_lookup(self.__ire, i)
        jreb = tf.nn.embedding_lookup(self.__ire, j)
        irbb = tf.nn.embedding_lookup(self.__irb, i)
        jrbb = tf.nn.embedding_lookup(self.__irb, j)
        iceb = tf.matmul(ic, self.__cem)
        jceb = tf.matmul(jc, self.__cem)

        x_ui  = tf.reduce_sum(tf.multiply(ureb, ireb)+tf.multiply(uceb, iceb), 1)
        x_uj  = tf.reduce_sum(tf.multiply(ureb, jreb)+tf.multiply(uceb, jceb), 1)
        x_uij = irbb - jrbb + x_ui - x_uj + tf.matmul(ic-jc, self.__icb)
        with tf.name_scope('output'):
            if self.mode == 'l2':
                self.obj = tf.reduce_sum(tf.log(1+tf.exp(-x_uij)))+\
                           0.5*tf.reduce_sum(self.__cem**2)*self.le+\
                           0.5*tf.reduce_sum((ureb**2+uceb**2)*self.lu+ireb**2*self.li+jreb**2*self.lj)+\
                           0.5*(tf.reduce_sum(irbb**2+jrbb**2)+tf.reduce_sum(self.__icb**2))*self.lb
            else:
                self.obj = tf.reduce_sum(tf.log(1+tf.exp(-x_uij)))+\
                           tf.reduce_sum(tf.abs(self.__cem))*self.le+\
                           tf.reduce_sum((tf.abs(ureb)+tf.abs(uceb))*self.lu+tf.abs(ireb)*self.li+tf.abs(jreb)*self.lj)+\
                           (tf.reduce_sum(tf.abs(irbb)+tf.abs(jrbb))+tf.reduce_sum(tf.abs(self.__icb)))*self.lb
        self.solver = tf.train.RMSPropOptimizer(self.lr).minimize(self.obj)
        return u, i, j, ic, jc

    def train(self, sampling: str = 'user uniform', epochs: int = 5, batch_size: int = 256, epoch_sample_limit: int = None, model_path: str = None) -> None:
        assert isinstance(sampling, str)
        assert isinstance(epochs, int)
        assert isinstance(batch_size, int)
        with tf.Graph().as_default():
            u, i, j, ic, jc = self.build_graph()
            self.__saver = tf.train.Saver()
            if epoch_sample_limit is not None:
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
                tprint('Training parameters: lu=%.6f, li=%.6f, lj=%.6f, lb=%.6f'%(self.lu, self.li, self.lj, self.lb))
                tprint('Learning rate is %.6f, regularization mode is %s'%(self.lr, self.mode))
                tprint('Training for %d epochs of %d batches using %s sampler'%(epochs, batch_limit, sampling))
                if self.fue is not None:
                    tprint('Initialize user embeddings')
                    self.__sess.run(tf.assign(self.__ure, self.fue[:, 0:self.k//2]))
                    self.__sess.run(tf.assign(self.__uce, self.fue[:, self.k//2:self.k]))
                if self.fie is not None:
                    tprint('Initialize item embeddings')
                    self.__sess.run(tf.assign(self.__ire, self.fie[:, 0:self.k//2]))
                if self.fib is not None:
                    tprint('Initialize item biases')
                    self.__sess.run(tf.assign(self.__irb, self.fib))
                for eid in range(epochs):
                    total_time = 0
                    bno = 1
                    for ub, ib, jb in sampler(batch_size):
                        t1 = time.time()
                        _, loss = self.__sess.run([self.solver, self.obj], feed_dict={u: ub, i: ib, j: jb, ic: self.feat[ib, :], jc: self.feat[jb, :]})
                        t2 = time.time()-t1
                        sys.stderr.write('\rEpoch=%3d, batch=%6d, loss=%8.2f, time=%4.4fs' % (eid+1, bno, loss, t2))
                        total_time += t2
                        bno += 1
                        if bno == batch_limit:
                            break
                    sys.stderr.write(' ... total time collapse %10.4fs'%(total_time))
                    sys.stderr.flush()
                    print()
            self.fue = self.__sess.run(tf.concat([self.__ure, self.__uce], 1))
            self.fie = self.__sess.run(tf.concat([self.__ire, tf.matmul(self.feat, self.__cem)], 1))
            self.fib = self.__sess.run(self.__irb + tf.matmul(self.feat, self.__icb))

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
