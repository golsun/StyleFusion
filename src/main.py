from shared import *
from tf_lib import *
from dataset import *
from model import *
#from dialog_gui import *
from classifier import load_classifier

"""
AUTHOR: Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""


def run_master(mode, args):

	if mode not in ['train','continue'] and args.restore != '':
		aa = args.restore.split('/')
		bb = []
		for a in aa:
			if len(a) > 0:
				bb.append(a)
		fld = '/'.join(bb[:-1])
		if mode in ['vali','vis','test']:
			vocab_only = False
			fld_data, _, _ = get_model_fld(args)
			path_bias_vocab = fld_data + '/vocab_bias.txt'
		else:
			vocab_only = True
			fld_data = fld
			path_bias_vocab = fld + '/vocab_bias.txt'
	else:
		vocab_only = False
		fld_data, fld_model, subfld = get_model_fld(args)
		fld = fld_model + '/' + subfld
		path_bias_vocab = fld_data + '/vocab_bias.txt'
	
	if os.path.exists(path_bias_vocab):
		allowed_words = [line.strip('\n').strip('\r') for line in open(path_bias_vocab, encoding='utf-8')]
	else:
		allowed_words = None

	model_class = args.model_class.lower()
	if model_class.startswith('fuse'):
		Master = StyleFusion
	elif model_class == 'mtask':
		Master = VanillaMTask
	elif model_class == 's2s':
		Master = Seq2Seq
	elif model_class == 'lm':
		Master = LanguageModel
	elif model_class == 's2s+lm':
		pass
	else:
		raise ValueError
	
	if model_class == 's2s+lm':
		master = Seq2SeqLM(args, allowed_words)
	else:
		dataset = Dataset(fld_data, 
			max_ctxt_len=args.max_ctxt_len, max_resp_len=args.max_resp_len,
			vocab_only=vocab_only, noisy_vocab=args.noisy_vocab)

		master = Master(dataset, fld, args, new=(mode=='train'), allowed_words=allowed_words)
		if mode != 'train':
			if args.restore.endswith('.npz') or model_class == 's2s+lm':
				restore_path = args.restore
			else:
				restore_path = master.fld + '/models/%s.npz'%args.restore
			master.load_weights(restore_path)

	if mode in ['vis', 'load']:
		return master

	CLF_NAMES = []
	print('loading classifiers '+str(CLF_NAMES))
	master.clf_names = CLF_NAMES
	master.classifiers = []
	for clf_name in CLF_NAMES:
		master.classifiers.append(load_classifier(clf_name))
	
	print('\n'+fld+'\n')
	if mode in ['continue', 'train']:
		
		ss = ['', mode + ' @ %i'%time.time()]
		for k in sorted(args.__dict__.keys()):
			ss.append('%s = %s'%(k, args.__dict__[k]))
		with open(master.fld + '/args.txt', 'a') as f:
			f.write('\n'.join(ss)+'\n')

		if args.debug:
			batch_per_load = 1
		else:
			if PHILLY:
				n_sample = 1280		# philly unstable for large memory 
			else:
				n_sample = 2560
			batch_per_load = int(n_sample/BATCH_SIZE)
		if mode == 'continue':
			master.vali()
		master.train(batch_per_load)

	elif 'summary' == mode:
		print(master.model.summary())

	elif mode in ['cmd', 'test', 'vali']:
		classifiers = []
		for clf_name in CLF_NAMES:
			classifiers.append(load_classifier(clf_name))

		if 'vali' == mode:		
			data = master.get_vali_data()
			s_decoded = eval_decoded(master, data, 
				classifiers=classifiers, corr_by_tgt=True, r_rand=args.r_rand,
				calc_retrieval=('holmes' in args.data_name.lower())
				)[0]
			s_surrogate = eval_surrogate(master, data)[0]
			print(restore_path)
			print()
			print(s_decoded)
			print()
			print(s_surrogate)
			return

		if model_class != 's2s+lm':
			with tf.variable_scope('base_rankder', reuse=tf.AUTO_REUSE):
				fld_base_ranker = 'restore/%s/%s/pretrained/'%(args.model_class.replace('fuse1','fuse'), args.data_name)
				dataset_base_ranker = Dataset(fld_base_ranker, 
					max_ctxt_len=args.max_ctxt_len, max_resp_len=args.max_resp_len,
					vocab_only=True, noisy_vocab=False)
				base_ranker = Master(dataset_base_ranker, fld_base_ranker, args, new=False, allowed_words=master.allowed_words)
				path = fld_base_ranker + '/' + open(fld_base_ranker+'/base_ranker.txt').readline().strip('\n')
				base_ranker.load_weights(path)
				print('*'*10 + ' base_ranker loaded from: '+path)
		else:
			base_ranker = None
		
		def print_results(results):
			ss = ['total', 'logP', 'rep', 'len', 's2s', 'clf_h', 'clf_v']
			print('; '.join([' '*(6-len(s))+s for s in ss]))
			for score, resp, terms in results:
				print('%6.3f; '%score + '; '.join(['%6.3f'%x for x in terms]) + '; ' + resp)

		if 'cmd' == mode:
			while True:
				print('\n---- please input ----')
				inp = input()
				infer_args = parse_infer_args()
				if inp == '':
					break
				results = infer_rank(inp, master, infer_args, base_ranker=base_ranker)
				print_results(results)


		elif 'test' == mode:
			path_in = DATA_PATH + '/test/' + args.test_fname
			if not PHILLY:
				fld_out = master.fld + '/eval2/'
			else:
				fld_out = OUT_PATH
			makedirs(fld_out)
			npz_name = args.restore.split('/')[-1].replace('.npz','')
			path_out = fld_out + '/' + args.test_fname + '_' + npz_name
			test_master(master, path_in, path_out, max_n_src=args.test_n_max, base_ranker=base_ranker, baseline=args.baseline, r_rand=args.r_rand)

	else:
		raise ValueError



def get_model_fld(args):
	data_name = args.data_name
	if PHILLY:
		data_name = data_name.replace('+','').replace('_','')
	fld_data = DATA_PATH +'/' + data_name

	master_config = 'width%s_depth%s'%(
			(args.token_embed_dim, args.rnn_units),
			(args.encoder_depth, args.decoder_depth))
	if args.max_ctxt_len != 90 or args.max_resp_len != 30:
		master_config += '_len' + str((args.max_ctxt_len, args.max_resp_len))

	master_config = master_config.replace("'",'')
	fld_model = OUT_PATH
	if args.debug:
		fld_model += '/debug'
	fld_model += '/' + args.data_name.replace('../','') + '_' + master_config

	subfld = []
	"""
	if args.randmix:
		s_mix = 'randmix'
		if args.ratio05 > 0:
			s_mix += '(0.5=%.2f)'%args.ratio05
	else:
	"""
	s_mix = 'mix'
	model_class = args.model_class.lower()
	if model_class == 's2s':
		subfld = ['s2s_%s(%.2f)'%(s_mix, args.conv_mix_ratio)]	# no conv data
	else:
		subfld = ['%s_%s(%.2f,%.2f)'%(model_class, s_mix, args.conv_mix_ratio, args.nonc_mix_ratio)]
	if args.noisy_vocab > 0:
		subfld.append('unk%.1fk'%(args.noisy_vocab/1000))
	if model_class.startswith('fuse'):
		subfld.append('std%.1f'%args.stddev)
		if args.reld:
			subfld.append('reld')

	subfld.append('lr'+str(args.lr))
	if len(args.fld_suffix) > 0:
		subfld.append(args.fld_suffix)
	subfld = '_'.join(subfld)

	return fld_data, fld_model.replace(' ',''), subfld.replace(' ','')



if __name__ == '__main__':
	parser.add_argument('mode')
	parser.add_argument('--skip', type=float, default=0.0)
	parser.add_argument('--test_fname', default='')
	parser.add_argument('--r_rand', '-r', type=float, default=-1)
	parser.add_argument('--test_n_max', '-n', type=int, default=2000)

	args = parser.parse_args()
	run_master(args.mode, args)




			