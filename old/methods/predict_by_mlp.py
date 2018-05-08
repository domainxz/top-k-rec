import cPickle as pickle
import argparse
import numpy as np
import timeit
import os,sys
import theano
import theano.tensor as T
from sklearn.cross_validation import KFold
from logistic_sgd import LogisticRegression

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):

        self.hiddenLayer1 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer1.output,
            n_in=n_hidden,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer2.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer1.W).sum()
            + abs(self.hiddenLayer2.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer1.W ** 2).sum()
            + (self.hiddenLayer2.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer1.params + self.hiddenLayer2.params + self.logRegressionLayer.params
        self.input = input
        self.output = self.logRegressionLayer.p_y_given_x;

def calculate_mean_and_std(X):
    means = np.mean(X, axis=0);
    stds  = np.std(X, axis=0);
    return means, stds;

def normailize_by_zvalue(means, stds, X):
    nz = (stds != 0);
    X -= means;
    X[:,nz] = np.divide(X[:,nz], stds[nz]);

def normailize_by_minmax(X):
    for i in range(X.shape[1]):
        minval = np.min(X[:,i]);
        maxval = np.max(X[:,i]);
        X[:,i] = (X[:,i] - minval) / (maxval - minval);

def main():
    parser = argparse.ArgumentParser(description="Transform csv files into numpy array");
    parser.add_argument('-d', '--data', required=True, help="The data directory");
    args = parser.parse_args();
    learning_rate = 0.0001
    L1_reg        = 0.00
    L2_reg        = 0.0001
    n_epochs      = 1000
    batch_size    = 32
    n_hidden      = 1000
    ds = pickle.load(open(os.path.join(args.data, 'ds.npy')));
    trI = ds['trI'];
    trX = ds['trX'].toarray();
    trY = ds['trY'].astype(np.int32);
    teI = ds['teI'];
    teX = ds['teX'].toarray();
    allX = np.vstack((trX, teX));
    means, stds = calculate_mean_and_std(allX);
    normailize_by_zvalue(means, stds, allX);
    #normailize_by_minmax(allX);
    trX = allX[0:trX.shape[0],:];
    teX = allX[trX.shape[0]:trX.shape[0]+teX.shape[0],:];
    kf = KFold(trX.shape[0], n_folds=5);
    trainIds = None;
    testIds  = None;
    for train, test in kf:
        trainIds = train;
        testIds  = test;
    cv_train_X = theano.shared(trX[trainIds,:], 'cv_train_X');
    cv_test_X  = theano.shared(trX[testIds,:], 'cv_test_X');
    cv_train_Y = theano.shared(trY[trainIds], 'cv_train_Y');
    cv_test_Y  = theano.shared(trY[testIds], 'cv_test_Y');
    ncvtr = len(trainIds);
    ncvte = len(testIds);
    nte   = teX.shape[0];
    n_train_batches = int(np.ceil(len(trainIds) * 1.0 / batch_size));
    n_valid_batches = int(np.ceil(len(testIds)  * 1.0 / batch_size));
    n_test_batches  = int(np.ceil(teX.shape[0]  * 1.0 / batch_size));
    teX = theano.shared(teX);
    rng = np.random.RandomState(1234);
    print('... building the model')
    left = T.lscalar()
    right = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=149,
        n_hidden=n_hidden,
        n_out=2
    )
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    validate_model = theano.function(
        inputs=[left,right],
        outputs=classifier.errors(y),
        givens={
            x: cv_test_X[left:right],
            y: cv_test_Y[left:right]
        }
    )
    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    train_model = theano.function(
        inputs=[left, right],
        outputs=cost,
        updates=updates,
        givens={
            x: cv_train_X[left:right],
            y: cv_train_Y[left:right]
        }
    )
    test_model = theano.function(
        inputs=[left, right],
        outputs=classifier.output,
        givens={
            x: teX[left:right],
        }
    )
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index * batch_size, min((minibatch_index + 1) * batch_size, ncvtr))
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i * batch_size, min((i+1) * batch_size, ncvte)) for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    probs = np.vstack([test_model(i * batch_size, min((i+1) * batch_size, nte)) for i in range(n_test_batches)]);
    fid = open('/local/db/uqdxingz/Santander/sub/mlp.csv', 'w');
    fid.write('ID,TARGET\n');
    for i in range(len(teI)):
        fid.write('%d,%.9f\n'%(int(teI[i]), probs[i][1]));
    fid.close();

if __name__ == '__main__':
    main();
