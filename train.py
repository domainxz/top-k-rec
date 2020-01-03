from single import *

model = BPR(k=50)
model.load_training_data('data/uid', 'data/vid', 'data/f0tr.txt')
# Training from scratch
model.train(epochs=5, batch_size=256, epoch_sample_limit=10e5)
model.export_embeddings('embed/bpr')
# Training from a pretrained model
model.train(epochs=5, batch_size=256, epoch_sample_limit=10e5, model_path='embed/bpr')

model = VBPR(k=50, d=20000)
model.load_training_data('data/uid', 'data/vid', 'data/f0tr.txt')
model.load_content_data('data/meta.pkl', 'data/vid')
model.train(epochs=5, batch_size=256, epoch_sample_limit=10e5)
model.export_embeddings('embed/vbpr')
model.train(epochs=5, batch_size=256, epoch_sample_limit=10e5, model_path='embed/vbpr')

model = WMF(k=50)
model.load_training_data('data/uid', 'data/vid', 'data/f0tr.txt')
model.train(max_iter=200, tol=1e-4)
model.export_embeddings('embed/wmf')
model.train(max_iter=20, model_path='embed/wmf')

model = CER(k=50, d=20000)
model.load_training_data('data/uid', 'data/vid', 'data/f0tr.txt')
model.load_content_data('data/meta.pkl', 'data/vid')
model.train(max_iter=20)
model.export_embeddings('embed/cer')
model.train(max_iter=20, model_path='embed/cer')

model = DPM(k = 50, d = 20000)
model.load_training_data('data/uid', 'data/vid', 'data/f0tr.txt')
model.load_content_data('data/meta.pkl', 'data/vid')
model.train(MLP, max_iter=20)
model.export_embeddings('embed/dpm')
model.train(MLP, max_iter=20, model_path='embed/dpm')
