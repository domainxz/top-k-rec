"""
    Matrix Factorization (MF) based on Bayesian Personalized Ranking (BPR)
    Sampling Method : uniform item sampling per user
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
    Reference       : "BPR : Bayesian Personalized Ranking from Implicit Feedback", Ste en Rendle, Christoph Freudenthaler, Zeno Gantner and Lars Schmidt-Thieme
"""

import theano, numpy
import theano.tensor as T
import time
import sys
from collections import defaultdict

class BPR(object):

    def __init__(self, K, users, items, lambda_u = 0.0025, lambda_i = 0.0025, lambda_j = 0.00025, lambda_bias = 0.0, learning_rate = 1.0e-4):
        self._K = K;
        self._train_users = users;
        self._train_items = items;
        self._n_users = len(users);
        self._n_items = len(items);
        self._lambda_u = lambda_u;
        self._lambda_i = lambda_i;
        self._lambda_j = lambda_j;
        self._lambda_bias = lambda_bias;
        self._learning_rate = learning_rate;
        self._train_dict = {};
        self._generate_train_model_function();

    def _generate_train_model_function(self):
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')
        
        self.W = theano.shared(0.01 * numpy.random.randn(self._n_users, self._K).astype('float32'), name='W');
        self.H = theano.shared(0.01 * numpy.random.randn(self._n_items, self._K).astype('float32'), name='H');
        self.B = theano.shared(numpy.zeros(self._n_items).astype('float32'), name='B')
        x_ui  = T.dot(self.W[u], self.H[i].T).diagonal();
        x_uj  = T.dot(self.W[u], self.H[j].T).diagonal();
        x_uij = self.B[i] - self.B[j] + x_ui - x_uj;

        obj = T.sum(
                T.log(T.nnet.sigmoid(x_uij)).sum() - \
                self._lambda_u * 0.5 * (self.W[u] ** 2).sum() - \
                self._lambda_i * 0.5 * (self.H[i] ** 2).sum() - \
                self._lambda_j * 0.5 * (self.H[j] ** 2).sum() - \
                self._lambda_bias * 0.5 * (self.B[i] ** 2 + self.B[j] ** 2).sum()
              )

        cost = -obj

        g_cost_W = T.grad(cost=cost, wrt=self.W)
        g_cost_H = T.grad(cost=cost, wrt=self.H)
        g_cost_B = T.grad(cost=cost, wrt=self.B)

        updates = [ 
                  (self.W, self.W - self._learning_rate * g_cost_W),
                  (self.H, self.H - self._learning_rate * g_cost_H),
                  (self.B, self.B - self._learning_rate * g_cost_B)
                  ]
        self.train_model = theano.function(inputs=[u,i,j], outputs=cost, updates=updates);

    def train(self, train_data, epochs=30, batch_size=256):
        if len(train_data) < batch_size:
            sys.stderr.write("WARNING: Batch size is greater than number of training samples, switching to a batch size of %s\n" % str(len(train_data)))
            batch_size = len(train_data)
        self._train_dict = self._data_to_dict(train_data, self._train_users, self._train_items)
        n_sgd_samples = len(train_data) * epochs;
        sgd_users, sgd_pos_items, sgd_neg_items = self._uniform_user_sampling(n_sgd_samples)
        z = 0
        t2 = t1 = t0 = time.time()
        while (z+1)*batch_size < n_sgd_samples:
            self.train_model(
                sgd_users[z*batch_size: (z+1)*batch_size],
                sgd_pos_items[z*batch_size: (z+1)*batch_size],
                sgd_neg_items[z*batch_size: (z+1)*batch_size]
            )
            z += 1
            t2 = time.time()
            sys.stderr.write("\rProcessed %s ( %.2f%% ) in %.4f seconds" %(str(z*batch_size), 100.0 * float(z*batch_size)/n_sgd_samples, t2 - t1))
            sys.stderr.flush()
            t1 = t2
        if n_sgd_samples > 0:
            sys.stderr.write("\nTotal training time %.2f seconds; %e per sample\n" % (t2 - t0, (t2 - t0)/n_sgd_samples))
            sys.stderr.flush()

    def _uniform_user_sampling(self, n_samples):
        sys.stderr.write("Generating %s random training samples\n" % str(n_samples))
        sgd_users = numpy.array(list(self._train_dict.keys()))[numpy.random.randint(len(self._train_dict), size=n_samples)]
        sgd_pos_items, sgd_neg_items = [], []
        for sgd_user in sgd_users:
            pos_item = self._train_dict[sgd_user][numpy.random.randint(len(self._train_dict[sgd_user]))]
            sgd_pos_items.append(pos_item)
            neg_item = numpy.random.randint(self._n_items)
            while neg_item in self._train_dict[sgd_user]:
                neg_item = numpy.random.randint(self._n_items)
            sgd_neg_items.append(neg_item)
        return sgd_users, sgd_pos_items, sgd_neg_items

    def _data_to_dict(self, data, users, items):
        data_dict = defaultdict(list)
        for (user, item) in data:
            data_dict[users[user]].append(items[item]);
        return data_dict
