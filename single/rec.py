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
from utils import get_id_dict_from_file, tprint, export_embed_to_file, get_embed_from_file


class REC(ABC):
    @abstractmethod
    def __init__(self):
        pass

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

    def export_embeddings(self, model_path: str) -> None:
        if os.path.exists(os.path.exists(model_path)):
            tprint('Saving model to path %s' % (model_path))
            if hasattr(self, 'fue'):
                export_embed_to_file(os.path.join(model_path, 'final-U.dat'), self.fue)
            if hasattr(self, 'fie'):
                export_embed_to_file(os.path.join(model_path, 'final-V.dat'), self.fie)
            if hasattr(self, 'fib'):
                export_embed_to_file(os.path.join(model_path, 'final-B.dat'), self.fib)

    def import_embeddings(self, model_path: str) -> None:
        file_path = os.path.join(model_path, 'final-U.dat')
        if os.path.exists(file_path):
            self.fue = get_embed_from_file(file_path, self.uids)
        file_path = os.path.join(model_path, 'final-V.dat')
        if os.path.exists(file_path):
            self.fie = get_embed_from_file(file_path, self.iids)
        file_path = os.path.join(model_path, 'final-B.dat')
        if os.path.exists(file_path):
            self.fib = get_embed_from_file(file_path, self.iids)
