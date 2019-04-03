from glob import glob
import os

MODEL_DIR = './models'
checkpoint_file = glob(os.path.join(MODEL_DIR, 'inception_resnet_v2_*.ckpt'))
print checkpoint_file

