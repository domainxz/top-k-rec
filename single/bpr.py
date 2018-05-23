"""
    Matrix Factorization (MF) based on Bayesian Personalized Ranking (BPR)
    Sampling Method : uniform item sampling per user
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
    Reference       : "BPR : Bayesian Personalized Ranking from Implicit Feedback", Ste en Rendle, Christoph Freudenthaler, Zeno Gantner and Lars Schmidt-Thieme
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import os
import sys
import tensorflow as tf
import time
from utils import get_train_data_from_file

class BPR(ABC):
    def __init__(self, k, lambda_u=2.5e-3, lambda_i=2.5e-3, lambda_j=2.5e-4, lambda_b=0, lr=1.0e-4, mode='l2'):
        self.k    = k;
        self.lu   = lambda_u;
        self.li   = lambda_i;
        self.lj   = lambda_j;
        self.lb   = lambda_b;
        self.lr   = lr;
        self.tf_config = tf.ConfigProto();
        self.tf_config.gpu_options.allow_growth=True;
        self.mode = mode;
    
    def load_training_data(self, training_file, data_copy=False):
        print ('Load training data from %s'%(training_file));
        self.data, self.tr_uids, self.tr_iids = get_train_data_from_file(training_file);
        self.epoch_sample_limit = len(self.data);
        self.n_users = len(self.tr_uids);
        self.n_items = len(self.tr_iids);
        self.tr_data = self._data_to_training_dict(self.data, self.tr_uids, self.tr_iids);
        self.tr_users = list(self.tr_data.keys());
        if not data_copy:
            del self.data;
        print('Loading finished!');

    def build_graph(self):
        with tf.variable_scope('bpr', reuse=tf.AUTO_REUSE):
            u = tf.placeholder(tf.int32, [None]);
            i = tf.placeholder(tf.int32, [None]);
            j = tf.placeholder(tf.int32, [None]);

            self.__ue = tf.get_variable(name="user_embed", shape=[self.n_users, self.k], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01));
            self.__ie = tf.get_variable(name="item_embed", shape=[self.n_items, self.k], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01));
            self.__ib = tf.get_variable(name="item_bias",  shape=[self.n_items],         dtype=tf.float32, initializer=tf.constant_initializer(0.0));
            
        ueb = tf.nn.embedding_lookup(self.__ue, u);
        ieb = tf.nn.embedding_lookup(self.__ie, i);
        jeb = tf.nn.embedding_lookup(self.__ie, j);
        ib  = tf.nn.embedding_lookup(self.__ib, i);
        jb  = tf.nn.embedding_lookup(self.__ib, j);

        x_ui  = tf.reduce_sum(tf.multiply(ueb, ieb), 1);
        x_uj  = tf.reduce_sum(tf.multiply(ueb, jeb), 1);
        x_uij = ib - jb + x_ui - x_uj;
        with tf.name_scope('output'):
            self.pred = tf.matmul(ueb, tf.transpose(ieb)) + ib;
            if self.mode == 'l2':
                self.obj = tf.reduce_sum(tf.log(1+tf.exp(-x_uij)))+\
                           0.5 * tf.reduce_sum(ueb**2*self.lu+ieb**2*self.li+jeb**2*self.lj)+\
                           0.5 * tf.reduce_sum(ib**2+jb**2)*self.lb;
            else:
                self.obj = tf.reduce_sum(tf.log(1+tf.exp(-x_uij)))+\
                           tf.reduce_sum(tf.abs(ueb)*self.lu+tf.abs(ieb)*self.li+tf.abs(jeb)*self.lj)+\
                           tf.reduce_sum(tf.abs(ib)+tf.abs(jb))*self.lb;
        self.solver = tf.train.RMSPropOptimizer(self.lr).minimize(self.obj);
        return u, i, j;

    def model_training(self, model_path, sampling='user uniform', epochs=10, batch_size=256):
        with tf.Graph().as_default():
            u, i, j = self.build_graph();
            batch_limit = self.epoch_sample_limit//batch_size + 1;
            sess = tf.Session(config=self.tf_config);
            sampler = None;
            if sampling == 'user uniform':
                sampler = self._uniform_user_sampling;
            with sess.as_default():
                sess.run(tf.global_variables_initializer());
                print ('Training parameters: lu=%.6f, li=%.6f, lj=%.6f, lb=%.6f'%(self.lu, self.li, self.lj, self.lb));
                print ('Learning rate is %.6f, regularization mode is %s'%(self.lr, self.mode));
                print ('Training for %d epochs of %d batches using %s sampler'%(epochs, batch_limit, sampling));
                for eid in range(epochs):
                    total_time = 0;
                    bno = 1;
                    for ub, ib, jb in sampler(batch_size):
                        t1 = time.time();
                        _, loss = sess.run([self.solver, self.obj], feed_dict={u:ub, i:ib, j:jb});
                        t2 = time.time()-t1;
                        sys.stderr.write('\rEpoch=%3d, batch=%6d, loss=%8.4f, time=%4.4fs'%(eid+1, bno, loss, t2));
                        total_time += t2;
                        bno += 1;
                        if bno == batch_limit:
                            break;
                    sys.stderr.write(' ... total time collapse %8.4fs'%(total_time));
                    sys.stderr.flush();
                    print();
            if os.path.exists(os.path.dirname(model_path)):
                print ('Saving model to path %s'%(model_path))
                saver = tf.train.Saver();
                saver.save(sess, model_path);
    
    def _uniform_user_sampling(self, batch_size): 
        ib = np.zeros(batch_size, dtype=np.int32);
        jb = np.zeros(batch_size, dtype=np.int32);
        while True:
            ub = np.random.choice(self.tr_users, batch_size);
            for i in range(batch_size):
                ib[i] = np.random.choice(self.tr_data[ub[i]]);
                jb[i] = np.random.choice(self.n_items);
                while jb[i] in self.tr_data[ub[i]]:
                    jb[i] = np.random.choice(self.n_items);
            yield ub, ib, jb;
    
    def _data_to_training_dict(self, data, users, items):
        data_dict = defaultdict(list)
        for (user, item) in data:
            data_dict[users[user]].append(items[item]);
        return data_dict;

