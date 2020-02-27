import os, random, sys, datetime, time, socket, io, h5py, argparse, shutil, io
import queue
import numpy as np
import scipy
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import matplotlib.pyplot as plt

"""
AUTHOR: 
Sean Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""

SOS_token = '_SOS_'
EOS_token = '_EOS_'
UNK_token = '_UNK_'

# just for arxiv
EQN_token = '_eqn_'
CITE_token = '_cite_'
IX_token = '_ix_'

hostname = socket.gethostname()
SKIP_VIS = hostname not in ['MININT-3LHNLKS', 'xiag-0228']
FIT_VERBOSE = 1

DATA_PATH = 'data/'
OUT_PATH = 'out/'

print('@'*20)
print('hostname:  %s'%hostname)
print('data_path: %s'%DATA_PATH)
print('out_path:  %s'%OUT_PATH)
print('@'*20)

PHILLY = False
BATCH_SIZE = 128#256

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='toy')
parser.add_argument('--batch_size', type=int, default=100)	
parser.add_argument('--max_n_trained', type=int, default=int(10e6))
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--token_embed_dim', type=int, default=1000)
parser.add_argument('--rnn_units', type=int, default=1000)
parser.add_argument('--encoder_depth', type=int, default=2)
parser.add_argument('--decoder_depth', type=int, default=2)
parser.add_argument('--stddev', type=float, default=0.1)
parser.add_argument('--wt_dist', type=float, default=1.)
parser.add_argument('--debug','-d', action='store_true')
parser.add_argument('--max_ctxt_len', type=int, default=90)	
parser.add_argument('--max_resp_len', type=int, default=30)	
parser.add_argument('--fld_suffix', type=str, default='')	
parser.add_argument('--conv_mix_ratio', type=float, default=0.0)	
parser.add_argument('--nonc_mix_ratio', type=float, default=1.0)
parser.add_argument('--clf_name', type=str, default='holmes')
parser.add_argument('--model_class', type=str, default='fuse')	
parser.add_argument('--restore', type=str, default='')
parser.add_argument('--noisy_vocab', type=int, default=-1)
parser.add_argument('--reld', action='store_true')
parser.add_argument('--ablation', '-abl', action='store_true')
parser.add_argument('--path_test', type=str)

if hostname == 'MININT-3LHNLKS':
	parser.add_argument('--cpu_only', '-c', action='store_true')

def reset_rand(RAND_SEED=9):
	random.seed(RAND_SEED)
	np.random.seed(RAND_SEED)

reset_rand()

def str2bool(s):
	return {'true':True, 'false':False}[s.lower()]

def strmap(x):
	if 'nan' in str(x):
		return 'nan'
	if isinstance(x, str):
		return x
	if int(x) == x:
		return '%i'%x
	return '%.4f'%x


def int2str(i):
	if i < 1000:
		return str(i)
	else:
		k = i/1000
		if int(k) == k:
			return '%ik'%k
		else:
			return '%.1fk'%k


def makedirs(fld):
	if not os.path.exists(fld):
		os.makedirs(fld)


def rand_latent(center, r, limit=True):
	if r == 0:
		return center

	noise = np.random.normal(size=center.shape)
	r_raw = np.sqrt(np.sum(np.power(noise, 2)))
	sampled = center + noise/r_raw*r
	if limit:
		return np.minimum(1, np.maximum(-1, sampled))
	else:
		return sampled


def calc_nltk_bleu(ref, hyp, max_ngram=4, smoothing_function=None):
	return sentence_bleu(
		[ref.split()], 
		hyp.split(), 
		weights=[1./max_ngram]*max_ngram,
		smoothing_function=smoothing_function,
		)


def calc_nltk_bleu_smoothed(ref, hyp, max_ngram=4):
	return calc_nltk_bleu(ref, hyp, max_ngram, 
		smoothing_function=SmoothingFunction().method7)

def euc_dist(a, b):
	# Euclidean distance
	if len(a.shape) == 1:
		return np.sqrt(np.sum(np.power(a - b, 2)))
	return np.sqrt(np.sum(np.power(a - b, 2), axis=1))


def now():
	return datetime.datetime.now()

