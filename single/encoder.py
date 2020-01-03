"""
    ENCODER root class
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
"""

from abc import ABC, abstractmethod
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


class ENCODER(ABC):
    @abstractmethod
    def out(self):
        pass

    @abstractmethod
    def fit(self):
        pass
