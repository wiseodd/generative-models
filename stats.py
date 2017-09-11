#!/usr/bin/env python3
import glob
import pandas as pd
import os
import sys
from pdb import set_trace as bp
import subprocess as sub

logdir='log'
if not os.path.exists(logdir):
        os.makedirs(logdir)

statfile = 'run.csv'

modeldir = []
modeldir.extend(glob.glob('GAN/*'))
modeldir.extend(glob.glob('RBM/*'))
modeldir.extend(glob.glob('VAE/*'))
print(modeldir)

df = pd.DataFrame(columns=['modeldir','familly','model','to_script','tf_script','n_script','is_to','is_tf'])

df['modeldir'] = modeldir
df['familly'],df['model'] = df.modeldir.str.split('/',1).str
for i, ival in enumerate(df.modeldir):
    to_script = glob.glob(ival+'/*_pytorch.py')
    tf_script = glob.glob(ival+'/*_tensorflow.py')
    if len(to_script) > 1:
        print("to many to_script",to_script)
        sys.exit(1)
    if len(tf_script) > 1:
        print("to many tf_script",tf_script)
        sys.exit(1)
    if len(tf_script) == 1:
        df['tf_script'].ix[i] = tf_script[0]
    if len(to_script) == 1:
        df['to_script'].ix[i] = to_script[0]
df.is_to = ~df.to_script.isnull()
df.is_tf = ~df.tf_script.isnull()
df.n_script = 1*df.is_to + 1*df.is_tf
