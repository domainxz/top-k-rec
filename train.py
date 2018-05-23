from single.bpr import BPR
from single.vbpr import VBPR

model = BPR(k=50);
model.load_training_data('data/f0tr.txt');
model.model_training(None);

model = VBPR(k=50, d=20000);
model.load_training_data('data/f0tr.txt');
model.load_content_data('data/meta.pkl', 'data/vid');
model.model_training('ckpt/vbpr.weights');
