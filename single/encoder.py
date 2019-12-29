"""
    ENCODER root class
    Author          : Xingzhong Du
    E-mail          : dxz.nju@gmail.com
"""

from abc import ABC, abstractmethod


class ENCODER(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def out(self):
        pass

    @abstractmethod
    def fit(self):
        pass
