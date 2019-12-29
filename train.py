from single.bpr import BPR
from single.vbpr import VBPR

from single.wmf import WMF
from single.dpm import DPM
from single.cer import CER

from single.mlp import MLP

model = BPR(k=50)
model.load_training_data('data/f0tr.txt', 'data/uid', 'data/vid')
model.train('embed/bpr', epochs=5, batch_size=256)

model = VBPR(k=50, d=20000)
model.load_training_data('data/f0tr.txt', 'data/uid', 'data/vid')
model.load_content_data('data/meta.pkl', 'data/vid')
model.train('embed/vbpr', epochs=5, batch_size=256)

model = WMF(k=50)
model.load_train_data('data/uid', 'data/vid', 'data/f0tr.txt')
model.train('embed/wmf', max_iter=200, tol=1e-4)

model = CER(k=50, d=20000)
model.load_train_data('data/uid', 'data/vid', 'data/f0tr.txt')
model.load_content_data('data/meta.pkl', 'data/vid')
model.train('embed/cer', max_iter=200)

model = DPM(k = 50, d = 20000)
model.load_train_data('data/uid', 'data/vid', 'data/f0tr.txt')
model.load_content_data('data/meta.pkl', 'data/vid')
model.train('embed/dpm', MLP, max_iter=200)
