import theano, numpy
import theano.tensor as T
import time
import sys
from collections import defaultdict

class RankingFusion(object):
    def __init__(self, scores, users, items, lambda_w = 0.0025, learning_rate = 1.0e-4):
        self._dim = scores.shape[2];
        self._train_users = users;
        self._train_items = items;
        self._n_users = len(users);
        self._n_items = len(items);
        self._lambda_w = lambda_w;
        self._learning_rate = learning_rate;
        self._train_dict = {};
        self._generate_train_model_function(scores);

    def _generate_train_model_function(self, scores):
       u = T.lvector('u')
       i = T.lvector('i')
       j = T.lvector('j')
       self.W = theano.shared(numpy.zeros((self._dim)).astype('float32'), name='W');
       self.S = theano.shared(scores, name='S');
       x_ui  = T.dot(self.W, self.S[u,i,:].T);
       x_uj  = T.dot(self.W, self.S[u,j,:].T);
       x_uij = x_ui - x_uj;
       obj = T.sum(
               T.log(T.nnet.sigmoid(x_uij)).sum() - \
               self._lambda_w * 0.5 * (self.W ** 2).sum()
               )
       cost = -obj
       g_cost_W = T.grad(cost=cost, wrt=self.W)
       updates = [
               (self.W, self.W - self._learning_rate * g_cost_W)
               ]
       self.train_model = theano.function(inputs=[u,i,j], outputs=cost, updates=updates);

    def train(self, train_data, epochs=30, batch_size=10000):
        if len(train_data) < batch_size:
            sys.stderr.write("WARNING: Batch size is greater than number of training samples, switching to a batch size of %s\n" % str(len(train_data)))
            batch_size = len(train_data)
        self._train_dict = self._data_to_dict(train_data, self._train_users, self._train_items)
        n_sgd_samples = 10000000;
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

