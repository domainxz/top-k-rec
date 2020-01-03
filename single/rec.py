"""
    REC class for hosting basic functions
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
"""

from abc import ABC, abstractmethod
import numpy as np
import os
import pickle
import scipy.sparse as ss
import tensorflow.compat.v1 as tf
from utils import get_id_dict_from_file, tprint, export_embed_to_file, get_embed_from_file

tf.disable_eager_execution()


class REC(ABC):
    @abstractmethod
    def load_training_data(self):
        pass

    def load_content_data(self, content_file: str, iid_file: str) -> None:
        tprint('Load content data from %s' % (content_file))
        fiids = get_id_dict_from_file(iid_file)
        self.feat = np.zeros((self.n_items, self.d), dtype=np.float32)
        feat = pickle.load(open(content_file, 'rb'), encoding='latin1')
        if ss.issparse(feat):
            feat = feat.toarray()
        for iid in self.iids:
            if iid in fiids:
                self.feat[self.iids[iid], :] = feat[fiids[iid], :]
        tprint('Loading finished!')

    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def export_model(self, model_path: str) -> None:
        pass

    def export_embeddings(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            tprint('%s does not exist, create it instead' % model_path)
            os.mkdir(model_path)
        if os.path.isdir(model_path):
            if hasattr(self, 'fue'):
                tprint('Saving user embeddings to %s' % os.path.join(model_path, 'final-U.dat'))
                export_embed_to_file(os.path.join(model_path, 'final-U.dat'), self.fue)
            if hasattr(self, 'fie'):
                tprint('Saving item embeddings to %s' % os.path.join(model_path, 'final-V.dat'))
                export_embed_to_file(os.path.join(model_path, 'final-V.dat'), self.fie)
            if hasattr(self, 'fib'):
                tprint('Saving item biases to %s' % os.path.join(model_path, 'final-B.dat'))
                export_embed_to_file(os.path.join(model_path, 'final-B.dat'), self.fib)
            self.export_model(model_path)
        else:
            tprint('%s is not a folder' % model_path)

    @abstractmethod
    def import_model(self, model_path: str) -> None:
        pass

    def import_embeddings(self, model_path: str) -> None:
        file_path = os.path.join(model_path, 'final-U.dat')
        if os.path.exists(file_path):
            tprint('Loading user embeddings from %s' % file_path)
            self.fue = get_embed_from_file(file_path, self.uids)
        file_path = os.path.join(model_path, 'final-V.dat')
        if os.path.exists(file_path):
            tprint('Loading item embeddings from %s' % file_path)
            self.fie = get_embed_from_file(file_path, self.iids)
        file_path = os.path.join(model_path, 'final-B.dat')
        if os.path.exists(file_path):
            tprint('Loading item biases from %s' % file_path)
            self.fib = get_embed_from_file(file_path, self.iids)
        self.import_model(model_path)
