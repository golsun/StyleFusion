from shared import *

"""
AUTHOR: 
Sean Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""


def load_vocab(path):
	with io.open(path, encoding='utf-8') as f:
		lines = f.readlines()

	index2token = dict()
	token2index = dict()
	for i, line in enumerate(lines):
		token = line.strip('\n').strip()
		index2token[i + 1] = token 			# start from 1, as 0 reserved for PAD
		token2index[token] = i + 1

	assert(SOS_token in token2index)
	assert(EOS_token in token2index)
	assert(UNK_token in token2index)
	return index2token, token2index


class Dataset:

	def __init__(self, 
		fld_data, 
		max_ctxt_len=93,
		max_resp_len=30,
		vocab_only=False,
		noisy_vocab=-1,
		noisy_AE_src=True,
		noisy_bias=True,	# whether add UNK noise to bias data (conv and nonc, src and tgt)
		):

		self.max_ctxt_len = max_ctxt_len
		self.max_resp_len = max_resp_len
		self.noisy_vocab = noisy_vocab
		self.noisy_AE_src = noisy_AE_src
		self.noisy_bias = noisy_bias
			
		types = ['base_conv','bias_conv', 'base_nonc', 'bias_nonc']

		self.fld_data = fld_data
		self.path_vocab = fld_data + '/vocab.txt'
		self.index2token, self.token2index = load_vocab(self.path_vocab)
		self.num_tokens = len(self.token2index)	# not including 0-th
		if self.noisy_vocab > 0:
			self.prob_keep = dict()
			for ix in self.index2token:
				self.prob_keep[ix] = np.exp(-ix/self.noisy_vocab)

		if vocab_only:
			return
	
		self.paths = dict()
		self.files = dict()
		self.n_reset = dict()
		for sub in ['train', 'vali', 'test']:
			self.paths[sub] = dict()
			self.files[sub] = dict()
			self.n_reset[sub] = dict()
			for tp in types:
				self.n_reset[sub][tp] = -1
				self.paths[sub][tp] = fld_data + '/%s_%s.num'%(tp, sub)
				self.reset(sub, tp)
		
		for k in self.files:
			print(k, self.files[k].keys())

	def reset(self, sub, tp=None):
		if tp is None:
			types = self.files[sub].keys()
		else:
			types = [tp]
		for tp in types:
			if os.path.exists(self.paths[sub][tp]):
				line = open(self.paths[sub][tp]).readline().strip('\n')
				if len(line) > 0:
					self.files[sub][tp] = open(self.paths[sub][tp])
			self.n_reset[sub][tp] += 1


	def seq2txt(self, seq):
		words = []
		for j in seq:
			if j == 0:		# skip PAD
				continue	
			words.append(self.index2token[int(j)])
		return ' '.join(words)


	def txt2seq(self, text):
		tokens = text.strip().split()
		seq = []
		for token in tokens:
			seq.append(self.token2index.get(token, self.token2index[UNK_token]))
		return seq


	def seqs2enc(self, seqs, max_len):
		inp = np.zeros((len(seqs), max_len))
		for i, seq in enumerate(seqs):
			for t in range(min(max_len, len(seq))):
				inp[i, t] = seq[t]
		return inp


	def seqs2dec(self, seqs, max_len):
		
		# len: +2 as will 1) add EOS and 2) shift to right by 1 time step
		# vocab: +1 as mask_zero (token_id == 0 means PAD)

		ix_SOS = self.token2index[SOS_token]
		ix_EOS = self.token2index[EOS_token]

		inp = np.zeros((len(seqs), max_len + 2))
		out = np.zeros((len(seqs), max_len + 2, self.num_tokens + 1))
		for i, seq in enumerate(seqs):
			seq = seq[:min(max_len, len(seq))]
			for t, token_index in enumerate(seq):
				inp[i, t + 1] = token_index		# shift 1 time step
				out[i, t, token_index] = 1.
			inp[i, 0] = ix_SOS 				# inp starts with EOS
			out[i, len(seq), ix_EOS] = 1.		# out ends with EOS

		return inp, out


	def skip(self, max_n, mix_ratio, conv_only=False):
		sub = 'train'
		if isinstance(mix_ratio, int) or isinstance(mix_ratio, float):
			mix_ratio = (mix_ratio,)
		
		def _read(tp, n, m):
			for _ in self.files[sub][tp]:
				if m >= n:
					break
				m += 1
				if m%1e5 == 0:
					print('%s skipped %.2f M'%(tp, m/1e6))
			return m

		m = dict()
		suffix = ['conv']
		if not conv_only:
			suffix.append('nonc')
		
		for i in range(len(suffix)):
			suf = suffix[i]
			for tp, n in [
				('base_'+suf, max_n * (1. - mix_ratio[i])), 
				('bias_'+suf, max_n * mix_ratio[i])
				]:
				m[tp] = 0
				if n < 1 or tp not in self.files[sub]:
					continue
				while m[tp] < n:
					m_ = _read(tp, n, m[tp])
					if m_ == m[tp]:
						self.reset(sub, tp)
					m[tp] = m_
					if m_ >= n:
						break

		print('conv skipped %.2f M'%((m['base_conv'] + m['bias_conv'])/1e6))
		if not conv_only:
			print('nonc skipped %.2f M'%((m['base_nonc'] + m['bias_nonc'])/1e6))


	def add_unk_noise(self, seqs):
		if self.noisy_vocab < 0 or len(seqs) == 0:
			return seqs
		ix_unk = self.token2index[UNK_token]
		ret = []
		n = 0
		old_n_unk = 0
		new_n_unk = 0
		for seq in seqs:
			noisy = []
			n += len(seq)
			for ix in seq:
				old_n_unk += (ix == ix_unk)
				if np.random.random() > self.prob_keep[ix]:
					noisy.append(ix_unk)
				else:
					noisy.append(ix)
				new_n_unk += (noisy[-1] == ix_unk)
			ret.append(noisy)
		print('unk increased from %.2f to %.2f'%(old_n_unk/n, new_n_unk/n))
		return ret


	def feed_data(self, sub, max_n, check_src=False, mix_ratio=(0.,0.), conv_only=False):
		if isinstance(mix_ratio, int) or isinstance(mix_ratio, float):
			mix_ratio = (mix_ratio,)
		print('loading data, check_src = %s, mix_ratio = %s'%(check_src, mix_ratio))

		# load conversation data -------------

		def _read_conv(tp, n, prev_ctxt, seqs):
			for line in self.files[sub][tp]:
				if len(seqs) >= n:
					break
				tt = line.strip('\n').split('\t')
				if len(tt) != 2:
					continue
				seq_ctxt, seq_resp = tt
				if check_src and (seq_ctxt == prev_ctxt):
					continue
				prev_ctxt = seq_ctxt
				seq_ctxt = [int(k) for k in seq_ctxt.split()]
				seq_resp = [int(k) for k in seq_resp.split()]
				seq_ctxt = seq_ctxt[-min(len(seq_ctxt), self.max_ctxt_len):]
				seq_resp = seq_resp[:min(len(seq_resp), self.max_resp_len)]

				seqs.append((seq_ctxt, seq_resp))
			return seqs, prev_ctxt

		# get conv from different tp 
		seqs = dict()
		for tp, n in [('base_conv', max_n * (1. - mix_ratio[0])), ('bias_conv', max_n * mix_ratio[0])]:
			seqs[tp] = []
			if n < 1 or tp not in self.files[sub]:
				continue
			prev_ctxt = ''
			while True:
				m = len(seqs[tp])
				seqs[tp], prev_ctxt = _read_conv(tp, n, prev_ctxt, seqs[tp])
				if len(seqs[tp]) >= n:
					break
				if len(seqs[tp]) == m:
					self.reset(sub, tp)
			print('conv from %s: %i/%i'%(tp, len(seqs[tp]), n))
		if 'bias_conv' in seqs and self.noisy_bias:
			seqs_ctxt = self.add_unk_noise([seq for seq, _ in seqs['bias_conv']])
			seqs_resp = self.add_unk_noise([seq for _, seq in seqs['bias_conv']])
			seqs['bias_conv'] = [(seqs_ctxt[i], seqs_resp[i]) for i in range(len(seqs['bias_conv']))]
				
		# then mix them
		ids = []
		for tp in seqs:
			ids += [(tp, i) for i in range(len(seqs[tp]))] 
		np.random.shuffle(ids)
		seqs_ctxt = []
		seqs_resp = []
		for tp, i in ids:
			seqs_ctxt.append(seqs[tp][i][0])
			seqs_resp.append(seqs[tp][i][1])

		inp_enc_ctxt = self.seqs2enc(seqs_ctxt, self.max_ctxt_len)
		if self.noisy_AE_src:
			inp_enc_resp = self.seqs2enc(self.add_unk_noise(seqs_resp), self.max_resp_len)
		else:
			inp_enc_resp = self.seqs2enc(seqs_resp, self.max_resp_len)
		inp_dec_resp, out_dec_resp = self.seqs2dec(seqs_resp, self.max_resp_len)

		n_sample_conv = len(ids)
		d_inp_enc = {'ctxt':inp_enc_ctxt, 'resp':inp_enc_resp}
		d_inp_dec = {'resp':inp_dec_resp}
		d_out_dec = {'resp':out_dec_resp}

		def get_ret(n, dd):
			n = BATCH_SIZE * int(n/BATCH_SIZE)
			ret = {'n_sample':n}
			for d_name in dd:
				d = dd[d_name]
				for k in d:
					if isinstance(d[k], list):
						d[k] = d[k][:n]
					else:
						d[k] = d[k][:n, :]
				ret[d_name] = d
			return ret

		if conv_only:
			return get_ret(n_sample_conv, {
				'inp_enc':d_inp_enc, 
				'inp_dec':d_inp_dec, 
				'out_dec':d_out_dec,
				'seqs':{'resp':seqs_resp},
				})

		# load non-conversation (nonc) data -------------

		def _read_nonc(tp, n, seqs):
			for line in self.files[sub][tp]:
				if len(seqs) >= n:
					break
				seq = [int(k) for k in line.strip('\n').split()]
				seq = seq[:min(len(seq), self.max_resp_len)]
				seqs.append(seq)
			return seqs

		# get nonc from different tp 
		seqs = dict()
		for tp, n in [('base_nonc', max_n * (1. - mix_ratio[1])), ('bias_nonc', max_n * mix_ratio[1])]:
			seqs[tp] = []
			if n < 1 or tp not in self.files[sub]:
				continue
			while True:
				m = len(seqs[tp])
				seqs[tp] = _read_nonc(tp, n, seqs[tp])
				if len(seqs[tp]) >= n:
					break
				if len(seqs[tp]) == m:
					self.reset(sub, tp)
			print('nonc from %s: %i/%i'%(tp, len(seqs[tp]), n))
		if 'bias_nonc' in seqs and self.noisy_bias:
			seqs['bias_nonc'] = self.add_unk_noise(seqs['bias_nonc'])

		seqs_nonc = seqs['base_nonc'] + seqs['bias_nonc']
		np.random.shuffle(seqs_nonc)
		
		if self.noisy_AE_src:
			inp_enc_nonc = self.seqs2enc(self.add_unk_noise(seqs_nonc), self.max_resp_len)
		else:
			inp_enc_nonc = self.seqs2enc(seqs_nonc, self.max_resp_len)
		inp_dec_nonc, out_dec_nonc = self.seqs2dec(seqs_nonc, self.max_resp_len)

		d_inp_enc['nonc'] = inp_enc_nonc
		d_inp_dec['nonc'] = inp_dec_nonc
		d_out_dec['nonc'] = out_dec_nonc
		n_sample = min(n_sample_conv, len(seqs_nonc))
		return get_ret(n_sample, {
				'inp_enc':d_inp_enc, 
				'inp_dec':d_inp_dec, 
				'out_dec':d_out_dec,
				'seqs':{'resp':seqs_resp, 'nonc':seqs_nonc},
				})

