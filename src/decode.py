from shared import *
from nltk.translate.bleu_score import SmoothingFunction
"""
AUTHOR: 
Sean Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""

class Decoder:

	def __init__(self, dataset, model, decoder_depth, latent_dim, allowed_words=None):
		self.dataset = dataset
		self.model = model
		self.decoder_depth = decoder_depth
		self.latent_dim = latent_dim

		if allowed_words is None:
			self.mask = np.array([1.] * (self.dataset.num_tokens + 1))
		else:
			self.mask = np.array([0.] * (self.dataset.num_tokens + 1))
			for word in allowed_words:
				ix = self._ix(word)
				if ix is not None:
					self.mask[ix] = 1.

		print('allowed words %i/%i'%(sum(self.mask), len(self.mask)))

		default_forbid = [UNK_token, '(', '__url__', ')', EQN_token, CITE_token, IX_token] #+ ['queer', 'holmes', 'sherlock', 'john', 'watson', 'bannister']
		for word in default_forbid:
			ix = self._ix(word)
			if ix is not None:
				self.mask[ix] = 0.		# in either case, UNK is not allowed


	def _ix(self, token):
		return self.dataset.token2index.get(token, None)


	def predict(self, latents, sampling=False, softmax_temperature=1, lm_wt=None):
		# autoregressive in parallel, greedy or softmax sampling
		
		latents = np.reshape(latents, (-1, self.latent_dim))	# (n, dim)
		n = latents.shape[0]
		n_vocab = len(self.mask)
		prev = np.zeros((n, 1)) + self._ix(SOS_token)
		states = [latents] * self.decoder_depth		# list of state, each is [n, dim]
		mask = np.repeat(np.reshape(self.mask, (1, -1)), n, axis=0)	# (n, vocab)
		logP = [0.] * n
		stop = [False] * n
		hyp = []
		for _ in range(n):
			hyp.append([])

		def sample_token_index_softmax(prob):
			if softmax_temperature != 1:
				prob = np.exp(np.log(prob) * softmax_temperature)
			return np.random.choice(n_vocab, 1, p=prob/sum(prob))[0]
		def sample_token_index_greedy(prob):
			return np.argmax(prob)
		if sampling:
			sample_token_index = sample_token_index_softmax
		else:
			sample_token_index = sample_token_index_greedy

		for _ in range(self.dataset.max_resp_len):
			out = self.model.predict([prev] + states)
			states = out[1:]
			tokens_proba = np.squeeze(out[0]) * mask	# squeeze: (n, 1, vocab) => (n, vocab)

			prev = [0] * n
			for i in range(n):
				if stop[i]:
					continue
				prob = tokens_proba[i,:].ravel()

				ix = sample_token_index(prob)
				logP[i] += np.log(prob[ix])
				hyp[i].append(ix)
				prev[i] = ix
				if ix == self._ix(EOS_token):
					stop[i] = True
			prev = np.reshape(prev, (n, 1))

		return [logP[i]/len(hyp[i]) for i in range(n)], hyp

	
	def evaluate(self, latents, tgt_seqs):
		# teacher-forcing in parallel

		latents = np.reshape(latents, (-1, self.latent_dim))	# (n, dim)
		n = latents.shape[0]
		states = [latents] * self.decoder_depth		# list of state, each is [n, dim]
		logP = [0.] * n
		prev = np.zeros((n, 1)) + self._ix(SOS_token)
		lens = [len(seq) for seq in tgt_seqs]
		epsilon = 1e-6

		for t in range(self.dataset.max_resp_len):
			out = self.model.predict([prev] + states)
			states = out[1:]
			tokens_proba = np.reshape(out[0], (n, -1))	# squeeze: (n, 1, vocab) => (n, vocab)
			prev = [0] * n
			for i in range(n):
				if t < lens[i]:
					ix = tgt_seqs[i][t]
					logP[i] += np.log(max(epsilon, tokens_proba[i, ix]))
					prev[i] = ix
			prev = np.reshape(prev, (n, 1))
				
		return [logP[i]/lens[i] for i in range(n)]
		#return [logP[i]/self.dataset.max_resp_len for i in range(n)]


	def predict_beam(self, latents, beam_width=10, n_child=3, max_n_hyp=100):
		# multi-head beam search, not yet parallel
		
		prev = np.atleast_2d([self._ix(SOS_token)])
		beam = []
		for latent in latents:
			latent = np.atleast_2d(latent)
			states = [latent] * self.decoder_depth
			node = {'states':states[:], 'prev':prev, 'logP':0, 'hyp':[]}
			beam.append(node)
		print('beam search initial n = %i'%len(beam))

		results = queue.PriorityQueue()
		t = 0
		while True:
			t += 1
			if t > 20:#self.dataset.max_tgt_len:
				break
			if len(beam) == 0:
				break

			pq = queue.PriorityQueue()
			for node in beam:
				out = self.model.predict([node['prev']] + node['states'])
				tokens_proba = out[0].ravel()
				states = out[1:]

				tokens_proba = tokens_proba * self.mask
				tokens_proba = tokens_proba/sum(tokens_proba)
				top_tokens = np.argsort(-tokens_proba)
				for ix in top_tokens[:n_child]:

					logP = node['logP'] + np.log(tokens_proba[ix])
					hyp = node['hyp'][:] + [ix]
					if ix == self._ix(EOS_token):
						results.put((logP/t, hyp))
						if results.qsize() > max_n_hyp:
							results.get()	# pop the hyp of lowest logP/t
						continue

					pq.put((
							logP, 	# no need to normalize to logP/t as every node is at the same t
							np.random.random(),	# to avoid the case logP is the same
							{
								'states':states,
								'prev':np.atleast_2d([ix]),
								'logP':logP,
								'hyp':hyp,
							}
						))
					if pq.qsize() > beam_width:
						pq.get()	# pop the node of lowest logP to maintain at most beam_width nodes => but this will encourage bland response

			beam = []
			while not pq.empty():
				_, _, node = pq.get()
				beam.append(node)

		logPs = []
		hyps = []
		while not results.empty():
			logP, hyp = results.get()
			logPs.append(logP)
			hyps.append(hyp)
		return logPs, hyps




def rank_nbest(hyps, logP, logP_center, master, inp, infer_args=dict(), base_ranker=None):
	# make sure hyps are list of str, and inp is str
	# as base_ranker, master, and clf may not share the same vocab

	assert(isinstance(hyps, list))
	assert(isinstance(hyps[0], str))
	assert(isinstance(inp, str))		
	
	hyps_no_ie = []
	for hyp in hyps[:]:
		hyps_no_ie.append((' '+hyp+' ').replace(' i . e . ,',' ').replace(' i . e. ',' ').strip())
	hyps = hyps_no_ie[:]

	wt_clf = infer_args.get('wt_clf', 0) / len(master.classifiers)
	wt_rep = infer_args.get('wt_rep', 0)
	wt_len = infer_args.get('wt_len', 0)
	wt_center = infer_args.get('wt_center', 0)
	wt_base = infer_args.get('wt_base', 0)
	
	n = len(logP)
	clf_score = []
	max_tgt_len = 30
	for clf in master.classifiers:
		clf_score.append(clf.predict(hyps).ravel())

	if base_ranker is not None:
		hyp_seqs_base = [base_ranker.dataset.txt2seq(hyp) for hyp in hyps]
		inp_seq_base = base_ranker.dataset.txt2seq(inp)
		latent_base = base_ranker.model_encoder['S2S'].predict(np.atleast_2d(inp_seq_base))
		logP_base = base_ranker.decoder.evaluate([latent_base]*n, hyp_seqs_base)
	else:
		logP_base = [0] * n
		
	pq = queue.PriorityQueue()
	for i in range(n):
		hyp = hyps[i]
		rep = repetition_penalty(hyp)
		l = min(max_tgt_len, len(hyp.split()))/max_tgt_len
		score = logP[i] + wt_center * logP_center[i] + wt_rep * rep + wt_len * l + wt_base * logP_base[i]

		clf_score_ = []
		for k in range(len(master.classifiers)):
			s = clf_score[k][i]
			score += wt_clf * s
			clf_score_.append(s)
		pq.put((-score, hyp, (logP[i], logP_center[i], logP_base[i], rep, l) + tuple(clf_score_)))

	results = []
	while not pq.empty():
		neg_score, hyp, terms = pq.get()
		#if len(set(['queer', 'holmes', 'sherlock', 'john', 'watson', 'bannister']) & set(hyp.split())) > 0:
		#	continue
		hyp = (' ' + hyp + ' ').replace(' to day ',' today ').replace(' to morrow ',' tomorrow ')#.replace('mr barker','')
		results.append((-neg_score, hyp, terms))
	return results


def repetition_penalty(hyp):
	# simplified from https://sunlamp.visualstudio.com/sunlamp/_git/sunlamp?path=%2Fsunlamp%2Fpython%2Fdynamic_decoder_custom.py&version=GBmaster
	# ratio of unique 1-gram
	ww = hyp.split()
	return np.log(min(1.0, len(set(ww)) / len(ww)))


def infer(latent, master, method='greedy', beam_width=10, n_rand=20, r_rand=1.5, softmax_temperature=1, lm_wt=0.5):
	if method == 'greedy':
		return master.decoder.predict(latent, lm_wt=lm_wt)
	elif method == 'softmax':
		return master.decoder.predict([latent] * n_rand, sampling=True, lm_wt=lm_wt)
	elif method == 'beam':
		return master.decoder.predict_beam([latent], beam_width=beam_width)
	elif method.startswith('latent'):
		latents = []
		if r_rand >= 0:
			rr = [r_rand] * n_rand
		else:
			rr = np.linspace(0, 5, n_rand)
		for r in rr:
			latents.append(rand_latent(latent, r, limit=True))
		if 'beam' in method:
			return master.decoder.predict_beam(latents, beam_width=beam_width)
		else:
			return master.decoder.predict(latents, sampling=('softmax' in method), softmax_temperature=softmax_temperature, lm_wt=lm_wt)
	else:
		raise ValueError

def infer_comb(inp, master):
	inp_seq = master.dataset.txt2seq(inp)
	latent = master.model_encoder['S2S'].predict(np.atleast_2d(inp_seq))
	reset_rand()
	logP, hyp_seqs = infer(latent, master, method='latent', n_rand=10, r_rand=-1)
	logP, hyp_seqs = remove_duplicate_unfished(logP, hyp_seqs, master.dataset.token2index[EOS_token])
	results = sorted(zip(logP, hyp_seqs), reverse=True)

	s = '-'*10 + '\n' + inp + '\n'
	for i, (logP, seq) in enumerate(results):
		hyp = master.dataset.seq2txt(seq)
		s += '%.3f'%logP + '\t' + hyp + '\n'
		if i == 4:
			break
	s += '-'*5 + '\n'
	return s


def remove_duplicate_unfished(logP, hyp_seqs, ix_EOS):
	d = dict()
	for i in range(len(logP)):
		k = tuple(hyp_seqs[i])
		if k[-1] != ix_EOS:
			continue
		if k not in d or logP[i] > d[k]:
			d[k] = logP[i]
	logP0, hyp0 = logP[0], hyp_seqs[0][:]
	logP = []
	hyp_seqs = []
	for k in d:
		logP.append(d[k])
		hyp_seqs.append(list(k))
	if len(logP) == 0:
		return [logP0], [hyp0]
	else:
		return logP, hyp_seqs


def parse_infer_args():
	arg = {'prefix':'S2S'}
	for line in open('src/infer_args.csv'):
		if line.startswith('#'):
			continue
		if ',' not in line:
			continue
		k, v = line.strip('\n').split(',')
		if k != 'method':
			if k in ['beam_width', 'n_rand']:
				v = int(v)
			else:
				v = float(v)
		arg[k] = v
	return arg


def infer_rank(inp, master, infer_args, base_ranker=None, unique=True, verbose=True):
	if verbose:
		print('infer_args = '+str(infer_args))
	inp_seq = master.dataset.txt2seq(inp)
	latent = master.model_encoder['S2S'].predict(np.atleast_2d(inp_seq))
	reset_rand()
	
	if verbose:
		print('infering...')
	t0 = datetime.datetime.now()
	logP, hyp_seqs = infer(latent, master, method=infer_args['method'], 
		beam_width=infer_args.get('beam_width'), n_rand=infer_args.get('n_rand'), r_rand=infer_args.get('r_rand'),
		softmax_temperature=infer_args.get('softmax_temperature'), lm_wt=infer_args.get('lm_wt'))
	t1 = datetime.datetime.now()
	if verbose:
		print('*'*10 + ' infer spent: '+str(t1-t0))

	n_raw = len(logP)
	logP, hyp_seqs = remove_duplicate_unfished(logP, hyp_seqs, master.dataset.token2index[EOS_token])
	if verbose:
		print('kept %i/%i after remove deuplication/unfisihed'%(len(logP), n_raw))
	
	hyps = [master.dataset.seq2txt(seq) for seq in hyp_seqs]
	if len(hyps) == 0:
		return []

	n_results = len(logP)
	if infer_args['method'] == 'latent' and infer_args['r_rand'] > 0:
		if verbose:
			print('calculating tf_logP...')
		logP_center = master.decoder.evaluate([latent]*n_results, hyp_seqs)
	else:
		logP_center = logP
		
	t2 = datetime.datetime.now()
	if verbose:
		print('*'*10 + ' logP_center spent: '+str(t2-t1))

	wts_classifier = []
	for clf_name in master.clf_names:
		wts_classifier.append(infer_args.get(clf_name, 0))

	if verbose:
		print('ranking...')
	results = rank_nbest(hyps, logP, logP_center, master, inp, infer_args, base_ranker)
	t3 = datetime.datetime.now()
	if verbose:
		print('*'*10 + ' ranking spent: '+str(t3-t2))
	return results

