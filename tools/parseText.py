import argparse
import cPickle as pickle
import nltk
import string
import os
import numpy as np
import re
import scipy.sparse as ss

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def main():
    parser = argparse.ArgumentParser(description="Parse texts from comments");
    parser.add_argument('-i', '--input', required=True, help="The input file where each line starts with a item id and its textual content, separated by the spliter");
    parser.add_argument('-o', '--output', required=True, help="The output path");
    parser.add_argument('-s', '--split', default="::", help="The spliter");
    parser.add_argument('-n', '--number', type=int, default=8000, help="The number of words to be used in the vectorization");
    args = parser.parse_args();
    # Initialize the parameters
    dictPath = os.path.join(args.output, 'dict.csv');
    mPath = os.path.join(args.output, 'multi.dat');
    nPath = os.path.join(args.output, 'mat.npy');
    tPath = os.path.join(args.output, 'tfidf.npy');
    itexts = dict();
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    # Read and process data from csv
    for line in open(args.input):
        terms = line.strip().split(args.split);
        iid    = int(terms[1]) - 1;
        text   = terms[3];
        if iid not in itexts:
            itexts[iid] = '';
        no_punctuation = text.decode('utf8').encode('ascii','ignore').lower().translate(replace_punctuation);
        no_punctuation = re.sub(r'\d+', '', no_punctuation);
        no_punctuation = ' '.join( [w for w in no_punctuation.split() if len(w)>1] )
        itexts[iid] += ' ' + no_punctuation;
    # Stem and generate the word list
    model = TfidfVectorizer(tokenizer=tokenize, stop_words='english', norm=None, use_idf=False);
    tfs   = model.fit_transform(itexts.values());
    vocabulary = model.vocabulary_;
    model = TfidfVectorizer(tokenizer=tokenize, stop_words='english', vocabulary=vocabulary, norm=None, use_idf=True);
    model.fit(itexts.values());
    idf   = model.idf_;
    wcounts  = np.sum(tfs.toarray().astype(np.int32), axis=0);
    wweights = dict();
    for word in vocabulary:
        wid = vocabulary[word];
        wweights[word] = wcounts[wid] * idf[wid];
    topwords = sorted(wweights, key=wweights.get, reverse=True)[0:8000];
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', vocabulary=topwords, use_idf=False, norm=None);
    tfs = tfidf.fit_transform(itexts.values()).toarray().astype(np.float32);
    fid = open(mPath, 'w');
    for i in range(len(itexts.keys())):
        count = np.sum(tfs[i,:]!=0);
        fid.write('%d'%count);
        if count != 0:
            for j in range(8000):
                if tfs[i,j] != 0:
                    fid.write(' %d:%d'%(j, int(tfs[i,j])));
        fid.write('\n');
    fid.close();
    pickle.dump(ss.csc_matrix(tfs), open(nPath, 'w'));
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', vocabulary=topwords, use_idf=True, norm='l2');
    tfs = tfidf.fit_transform(itexts.values()).toarray().astype(np.float32);
    pickle.dump(ss.csc_matrix(tfs), open(tPath, 'w'));
    fid = open(dictPath, 'w');
    for i in range(len(topwords)):
        fid.write(topwords[i] + '\n');
    fid.close();

if __name__ == '__main__':
    main();
