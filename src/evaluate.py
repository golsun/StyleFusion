from decode import *
from collections import defaultdict

"""
AUTHOR: 
Sean Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""

my_smooth = SmoothingFunction(epsilon=0.01).method1

def calc_entropy(seqs):
	etp_score = [0.0] * 4
	counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
	i = 0
	for words in seqs:
		i += 1
		for n in range(4):
			for idx in range(len(words)-n):
				ngram = tuple(words[idx:idx+n+1])
				counter[n][ngram] += 1

	for n in range(4):
		total = sum(counter[n].values())+ 1e-9
		for v in counter[n].values():
			etp_score[n] += - 1.0 * v /total * (np.log(v) - np.log(total))
	return etp_score


def calc_distinct(seqs):
	tokens = [0.0,0.0]
	types = [defaultdict(int),defaultdict(int)]
	for words in seqs:
		for n in range(2):
			for idx in range(len(words)-n):
				ngram = ' '.join(words[idx:idx+n+1])
				types[n][ngram] = 1
				tokens[n] += 1
	div1 = 1. * len(types[0].keys())/tokens[0]
	div2 = 1. * len(types[1].keys())/tokens[1]
	return [div1, div2]


def calc_retrieval_rate(hyps, refs, sim_type='01'):
	def calc_sim_01(hyp, ref):
		return int(tuple(hyp) == tuple(ref))

	if sim_type == '01':
		crit_sim = 0.5
		calc_sim = calc_sim_01
	else:
		raise ValueError

	m = 0
	n = 0
	for i, hyp in enumerate(hyps):
		if i%100 == 0 and i > 0:
			print('processed %i hyps, so far retrieval rate = %.3f'%(i, m/n))
		n += len(hyp)
		for ref in refs:
			sim = calc_sim(hyp, ref)
			if sim > crit_sim:
				m += len(hyp)
				break
		if i == 300:
			break

	return m/n
				

def get_available(master, data):
	n = data['base']['n_sample']
	available = [('base','conv'),('base','nonc')]
	if data['bias']['n_sample'] > 0:
		assert(n == data['bias']['n_sample'])
		available.append(('bias','nonc'))
		if master.bias_conv:
			available.append(('bias','conv'))
	return n, available



def eval_decoded(master, data, classifiers=[], corr_by_tgt=True, calc_retrieval=False, r_rand=0):
	return ''




def eval_surrogate(master, data):
	_, available = get_available(master, data)
	ss = [] 
	print('>'*10 + ' running eval_surrogate...')
	t00 = now()

	# decoder loss on pure data ------------------------

	ss = []
	logP_pure = dict()
	t0 = now()
	for tp in ['base','bias']:
		for prefix, inp, out, conv in [('S2S','ctxt','resp','conv'), ('AE','resp','resp','conv'), ('AE','nonc','nonc','nonc')]:
			if not (tp, conv) in available:
				continue
			if prefix not in master.prefix:
				continue
			k = (prefix, tp, out)
			logP_pure[k] = master.model_tf[prefix].evaluate(	# TODO: consistant with decoder.evaluate
				[data[tp]['inp_enc'][inp], data[tp]['inp_dec'][out]], 
				data[tp]['out_dec'][out],
				verbose=0)
			ss.append('logP(%s,%s,%s) = %.3f'%(k + (logP_pure[k],)))
	print('>'*10 + ' dec_pure spent '+str(now() - t0))

	if 'AE' not in master.prefix:
		return '\n'.join(ss), logP_pure, dict()
	
	# all loss terms on mixed data ------------------------

	t0 = now()
	_, inputs, outputs = master._inp_out_data(data['mix'])
	loss_mix = master.model.evaluate(inputs, outputs, verbose=0, batch_size=BATCH_SIZE)

	if master.name == 'fuse':
		ss += [
			'',
			'logP(S2S):             %.4f'%loss_mix[1],
			'logP(S2S,AE_resp):     %.4f'%loss_mix[2],
			'logP(AE_resp,AE_nonc): %.4f'%loss_mix[3],
			'rel_dist:              %.4f'%loss_mix[4],
			#'d(S2S,AE_resp): %.4f'%loss_mix[4],
			#'d(AE_resp,AE_nonc): %.4f'%loss_mix[5],
			#'d(S2S):     %.4f'%loss_mix[6],
			#'d(AE_resp): %.4f'%loss_mix[7],
			#'d(AE_nonc): %.4f'%loss_mix[8],
			]
	elif master.name == 'mtask':
		ss += [
			'',
			'logP(S2S):    %.4f'%loss_mix[1],
			'logP(AE_resp):%.4f'%loss_mix[2],
			'logP(AE_nonc):%.4f'%loss_mix[3],
			]
	
	print('>'*10 + ' full_cross spent '+str(now() - t0))
	print('>'*10 + ' total spent '+str(now() - t00))
	return '\n'.join(ss), logP_pure, loss_mix



def test_master(master, path_in, path_out, max_n_src=-1, base_ranker=None, baseline='', r_rand=-1):
	infer_args = parse_infer_args()
	if r_rand >= 0:
		infer_args['r_rand'] = r_rand
	print(infer_args)

	if master.name == 's2s+lm':
		path_out += '.lm%.2f'%infer_args['lm_wt']

	if baseline == '':
		if infer_args['method'] == 'latent':
			path_out += '_r%.1f'%infer_args['r_rand']
		elif infer_args['method'] == 'beam':
			path_out += '_beam%i'%infer_args['beam_width']
		else:
			path_out += '_'+ infer_args['method']
	else:
		if baseline == 'human':
			path_out += '_human'
		elif baseline in ['IR','rand']:
			path_out += '_retrieval'
	path_out += '.tsv'
	if baseline == 'rand' and os.path.exists(path_out):
		hide_bad(path_out, is_rand=True)
		return

	header = '\t'.join([
			'ix_src', 'type', 'txt', 'n_ref',
			'logP','logP_center','logP_base','rep','len',
			'neural','ngram',
			'bleu1','bleu2','bleu3','bleu4','best_ref',
			's1bleu1','s1bleu2','s1bleu3','s1bleu4','best_ref_s1bleu',
			])

	print(path_out)
	with open(path_out, 'w', encoding='utf-8') as f:
		f.write(header + '\n')
	prev_src = None
	refs_txt = []
	ix_src = -1
	lines = []
	for line in open(path_in, encoding='utf-8'):
		new_src_txt, new_ref_txt = line.strip('\n').split('\t')[:2]
		if new_src_txt == prev_src :
			refs_txt.append(new_ref_txt)
		else:
			if prev_src is not None:
				print('computing for src %i, which has %i refs'%(ix_src, len(refs_txt)))
				lines.append('\t'.join([str(ix_src), 'src.', prev_src]))
				lines += ['\t'.join([str(ix_src), 'ref%i'%i, ref_txt]) for i, ref_txt in enumerate(refs_txt)]
				lines.append('\n')
				if baseline == '':
					results = infer_rank(prev_src, master, infer_args, verbose=False, base_ranker=base_ranker)
				else:
					results = get_baseline_nbest(baseline, master, refs_txt, base_ranker, prev_src, infer_args)
				for i, (_, hyp_txt, terms) in enumerate(results):
					if i == infer_args['n_rand']:
						break
					ss = [str(ix_src), 'hyp%i'%i, hyp_txt[:], str(len(refs_txt))] + ['%.5f'% s for s in list(terms)]
					hyp_txt = hyp_txt.replace(EOS_token,'').strip()
					for smooth in [None, my_smooth]:
						bleu = [0.] * 4
						for ref_txt in refs_txt:
							if baseline == 'human' and hyp_txt.lower() == ref_txt.lower():
								continue
							for ngram in [1,2,3,4]:
								b = sentence_bleu([ref_txt.split()], hyp_txt.split(), 
										weights=[1./ngram]*ngram, smoothing_function=smooth)
								if b >= bleu[ngram-1]:
									if ngram == 2:
										best_ref = ref_txt
									bleu[ngram-1] = b
						ss += ['%.5f'% s for s in bleu] + [best_ref]
					lines.append('\t'.join(ss))
				with open(path_out, 'a', encoding='utf-8') as f:
					f.write('\n'.join(lines) + '\n')
				lines = ['\n']

			ix_src += 1
			prev_src = new_src_txt
			refs_txt = [new_ref_txt]
			if max_n_src == ix_src:
				break
	hide_bad(path_out, is_rand=(baseline in ['human','rand']))
		

def hide_bad(path_in, n_kept=1, is_rand=False, cut_by='logP_base', 
	use_clf='avg', method='+', src1turn=True, no_url=False, wt_style=0.5,
	style_cap=None, crit_rep=-0.3, crit_logP=-2.5):

	path_out = path_in[:]
	if not is_rand:
		if wt_style != 0.5:
			path_out += '.swt%.2f'%wt_style
		path_out += '.rep%.2f'%crit_rep
		if style_cap is not None:
			path_out += '.scap%.2f'%style_cap
		if wt_style < 1:
			if method == '+':
				path_out += '.%s%.2f+log_clf'%(cut_by,crit_logP)
			else:
				raise ValueError
			path_out += '.%s'%use_clf
	else:
		path_out += '.rand'
	if src1turn:
		path_out += '.src1turn'
	if no_url:
		path_out += '.no_url'
	path_out += '.tsv'

	f = open(path_in, encoding='utf-8')
	header = f.readline().strip('\n').split('\t')
	with open(path_out,'w',encoding='utf-8') as fout:
		fout.write('\t'.join(header)+'\n')
		fout.write(path_out+'\n')

	def _v(cells, k):
		v = cells[header.index(k)]
		if k in ['type','txt','best_ref']:
			return v
		if k == 'ix_src':
			return int(v)
		else:
			return float(v)

	#np.random.seed(8)
	def _write_lines(src_line, kept):
		if kept is None or len(kept) == 0:
			return
		if is_rand:
			np.random.shuffle(kept)
		else:
			kept = sorted(kept, reverse=True)
		"""
		print(src_line)
		for score, line in kept:
			print('%.4f\t%s'%(score, line))
		exit()
		"""
		kept = kept[:min(len(kept), n_kept)]
		lines = ['\n',src_line] + [line for _, line in kept]
		with open(path_out, 'a', encoding='utf-8') as fout:
			fout.write('\n'.join(lines).strip('\n')+'\n')

	src_line = None
	kept = None
	src_line = ''
	good_src = False
	for line in f:
		line = line.strip('\n')
		if len(line.strip('\t')) == 0:
			continue
		cells = line.strip('\n').split('\t')
		try:
			tp = _v(cells, 'type')
		except IndexError:
			print(line)
			exit()
		if tp.startswith('src'):
			if good_src:
				_write_lines(src_line, kept)
			src_line = line.strip('\n')
			kept = []
			good_src = True
			src = _v(cells, 'txt')
			if src1turn and len(src.split(' EOS ')) > 1:
				good_src = False
			if no_url and '__url__' in src:
				good_src = False
		elif tp.startswith('hyp'):
			if not good_src:
				continue
			if is_rand:
				kept.append((None, line))
			else:
				if _v(cells, 'rep') < crit_rep or _v(cells, cut_by) < crit_logP:
					score = -1e3 + _v(cells, cut_by)
				else:
					ngram = _v(cells, 'ngram')
					neural = _v(cells, 'neural')
					if style_cap is not None:
						ngram = min(style_cap, ngram)
						neural = min(style_cap, neural)
						
					if use_clf == 'avg':
						style = 0.5 * (ngram + neural)
					elif use_clf == 'min':
						style = min(neural, ngram)
					elif use_clf == 'neural':
						style = neural
					elif use_clf == 'ngram':
						style = ngram
					prob =  np.exp(_v(cells, cut_by))
					if method == 'cut':
						if _v(cells, cut_by) < infer_args['crit_logP']:
							score = -1e3
						else:
							score = style
					elif method == 'f1':
						#style = min(f1_cap, style)
						score = style * prob / ((1 - f1_prob_wt) * style + f1_prob_wt * prob)
					elif method == '+':
						score = (1. - wt_style) * prob + wt_style * style
					elif method == 'x':
						score = prob * (1. - wt_style + wt_style * style)
					else:
						raise ValueError
				kept.append((score, line.strip('\n')))
	if good_src:
		_write_lines(src_line, kept)
	return path_out


def get_baseline_nbest(baseline, master, refs_txt, base_ranker, inp, infer_args):
	if baseline in ['IR','rand'] and not hasattr(master, 'bias_nonc_txts'):
		print('>'*10 + ' Building stylized texts dataset...')
		ll = []
		master.bias_nonc_txts = []
		for line in open(master.dataset.paths['train']['bias_nonc']):
			seq = line.strip('\n').split()
			if len(seq) >= 30:
				continue
			txt = master.dataset.seq2txt([int(x) for x in seq])
			if UNK_token in txt:	# to be consistent with master.decoder.predict
				continue
			ll.append(len(seq))
			master.bias_nonc_txts.append(txt)
			if len(master.bias_nonc_txts) == 1e4:
				break
		print('>'*10 + ' avg_len = %.2f'%np.mean(ll))
	
	if baseline == 'human':			# retrieval from ref
		hyps = refs_txt
	elif baseline in ['rand','IR']: # retrieval from stylized texts
		ii = np.random.choice(len(master.bias_nonc_txts), infer_args['n_rand'], replace=False)
		hyps = [master.bias_nonc_txts[i] for i in ii]
	else:
		raise ValueError

	hyp_seqs = [master.dataset.txt2seq(hyp) for hyp in hyps]
	inp_seq = master.dataset.txt2seq(inp)
	latent = master.model_encoder['S2S'].predict(np.atleast_2d(inp_seq))
	logP = master.decoder.evaluate([latent]*len(hyps), hyp_seqs)

	return rank_nbest(hyps, logP, logP, master, inp, infer_args, base_ranker)


def calc_file_diversity(path):
	print(path)
	f = open(path, encoding='utf-8')
	header = f.readline().strip('\n').split('\t')
	ix_type = header.index('type')
	ix_txt = header.index('txt')
	hyps = []
	for line in f:
		cells = line.strip('\n').split('\t')
		if len(cells) < ix_type + 1:
			continue
		if cells[ix_type].startswith('hyp'):
			hyps.append(cells[ix_txt].split())
	return calc_entropy(hyps) + calc_distinct(hyps)