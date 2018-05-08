import numpy as np
import lda
import cPickle as pickle
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate the beta abd theta files after latent Dirichlet allocation (LDA) process.");
    parser.add_argument('-i', '--input', required=True, help="The input file where each line starts with the number of word as well as the sparse representation of word distribution");
    parser.add_argument('-o', '--output', required=True, help="The output path");
    args  = parser.parse_args();
    tfidf = pickle.load(open(args.input));
    feat  = tfidf.toarray().astype(np.int64);
    model = lda.LDA(n_topics=50, n_iter=1500, random_state=2017);
    model.fit(feat);
    fid   = open(os.path.join(args.output, 'init.beta'), 'w');
    beta  = model.topic_word_;
    for row in range(beta.shape[0]):
        fid.write('%f'%beta[row,0]);
        for col in range(1, beta.shape[1]):
            fid.write(' %f'%beta[row,col]);
        fid.write('\n');
    fid.close();
    fid   = open(os.path.join(args.output, 'init.theta'), 'w');
    theta = model.doc_topic_
    for row in range(theta.shape[0]):
        fid.write('%f'%theta[row,0]);
        for col in range(1, theta.shape[1]):
            fid.write(' %f'%theta[row,col]);
        fid.write('\n');
    fid.close();

if __name__ == '__main__':
    main();
