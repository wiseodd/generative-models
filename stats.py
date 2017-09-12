#!/usr/bin/env python3
import glob
import pandas as pd
import os
import sys
from pdb import set_trace as bp
import subprocess as sub
import re

logdir='log'
if not os.path.exists(logdir):
        os.makedirs(logdir)

scriptout = 'scripts.csv'
statout = 'stats.csv'

modeldir = []
modeldir.extend(glob.glob('GAN/*'))
modeldir.extend(glob.glob('RBM/*'))
modeldir.extend(glob.glob('VAE/*'))
print(modeldir)

df1 = pd.DataFrame(columns=['modeldir','familly','model','to_script','tf_script','n_script','is_to','is_tf'])

df1['modeldir'] = modeldir
df1['familly'],df1['model'] = df1.modeldir.str.split('/',1).str
for i, ival in enumerate(df1.modeldir):
    to_script = glob.glob(ival+'/*_pytorch.py')
    tf_script = glob.glob(ival+'/*_tensorflow.py')
    if len(to_script) > 1:
        print("to many to_script",to_script)
        sys.exit(1)
    if len(tf_script) > 1:
        print("to many tf_script",tf_script)
        sys.exit(1)
    if len(tf_script) == 1:
        df1['tf_script'].ix[i] = tf_script[0]
    if len(to_script) == 1:
        df1['to_script'].ix[i] = to_script[0]
df1.is_to = ~df1.to_script.isnull()
df1.is_tf = ~df1.tf_script.isnull()
df1.n_script = 1*df1.is_to + 1*df1.is_tf
df1.to_csv(scriptout,index=False)

logfiles=[]
logfiles.extend(glob.glob('log/*'))
df2 = pd.DataFrame()
df2['rplogfile'] = logfiles
df2['logfile'] = df2.rplogfile.str.split('log/').str[1]
df2 = pd.concat([df2, df2.logfile.str.split('.', expand=True)],axis=1)
df2['script']=df2[0]+'.'+df2[1]
df2.drop([0,1],axis=1,inplace=True)
df2['gpu'] = df2[2]
df2.drop(2,axis=1,inplace=True)
df2.drop(4,axis=1,inplace=True)
df2['date'] = df2[3]
df2.drop(3,axis=1,inplace=True)
df2['delta'] = None
df2['framework'] = None
from datetime import datetime
for i, ival in enumerate(df2.rplogfile):
        pat = r'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]'
        with open(ival,'r') as fin:
                lines = fin.readlines()
        t1 = re.search(pat, lines[0]).group(0)
        t1 = datetime.strptime(t1, '%Y-%m-%d %H:%M:%S')
        t2 = re.search(pat, lines[-1]).group(0)
        t2 = datetime.strptime(t2, '%Y-%m-%d %H:%M:%S')
        delta = t2 - t1
        df2['delta'].ix[i] = delta
df2['framework'][df2.script.str.match(r'.*pytorch.*')] = 'torch'
df2['framework'][df2.script.str.match(r'.*tensorflow.*')] = 'tensorflow'
df2.sort_values(['script','framework'],inplace=True)
df2.reset_index(inplace=True)
df2.to_csv(statout,index=False)
