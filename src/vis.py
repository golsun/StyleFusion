from shared import *
from tf_lib import *
from main import run_master, get_model_fld
from scipy.optimize import fmin_powell as fmin
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import scipy

"""
AUTHOR: 
Sean Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""



	

def dist_mat(coord):
	n = coord.shape[0]
	dist_T2T = np.zeros((n, n))
	for i in range(n):
		for j in range(i + 1, n):
			d = euc_dist(coord[i, :], coord[j, :])
			dist_T2T[i, j] = d
			dist_T2T[j, i] = d
	return dist_T2T




def interp(master, model_name, fld_save, type_='resp'):

	n = 1000
	print('building data...')
	_, d_inp_enc, d_inp_dec, d_out_dec, _ = master.dataset.feed_data('test', max_n=n, check_src=True)
	if type_ == 'resp':
		vec_u0 = master.model_encoder['S2S'].predict(d_inp_enc['ctxt'])
		vec_u1 = master.model_encoder['AE'].predict(d_inp_enc['resp'])
	elif type_ == 'stry':
		vec_u0 = master.model_encoder['AE'].predict(d_inp_enc['resp'])
		vec_u1 = master.model_encoder['AE'].predict(d_inp_enc['stry'])
	else:
		raise ValueError

	print('evaluating...')
	uu = np.linspace(0, 1, 11)
	NLL = []
	for u in uu:
		latent = vec_u0 + u * np.ones(vec_u0.shape) * (vec_u1 - vec_u0)
		NLL_resp = master.model_decoder_tf.evaluate(
					[latent, d_inp_dec['resp']], 
					d_out_dec['resp'],
					verbose=0)
		if type_ == 'resp':
			NLL_ = NLL_resp
		else:
			NLL_stry = master.model_decoder_tf.evaluate(
					[latent, d_inp_dec['stry']], 
					d_out_dec['stry'],
					verbose=0)
			NLL_ = NLL_resp * (1. - u) + u * NLL_stry 
		print('u = %.3f, NLL = %.3f'%(u, NLL_))
		NLL.append(NLL_)
	
	fig = plt.figure(figsize=(6,3))
	ax = fig.add_subplot(111)
	ax.plot(uu, NLL,'k.-')
	print(uu)
	print(NLL)
	ax.plot(0, NLL[0], 'ro')
	ax.plot(1, NLL[-1], 'bo')

	ax.text(0, NLL[0] + 0.5, '  '+r'$S$', color='r')
	ax.text(1, NLL[-1], '  '+r'$T$', color='b')

	plt.xlabel(r'$u$')
	plt.ylabel('NLL')
	plt.title(model_name+'\nNLL of interpolation: '+r'$S+u(T-S)$')
	plt.subplots_adjust(top=0.8)
	plt.subplots_adjust(bottom=0.2)
	plt.savefig(fld_save+'/interp_%s.png'%type_)

	with open(fld_save+'/interp_%s.tsv'%type_,'w') as f:
		f.write('\t'.join(['u'] + ['%.3f'%u for u in uu])+'\n')
		f.write('\t'.join(['NLL'] + ['%.3f'%l for l in NLL])+'\n')

	plt.show()






def clusters(master, model_name, fld_save, D=2, use_bias=True, n_batch=1):

	n_sample = BATCH_SIZE * n_batch
	method = 'MDS'
	#method = 'tSNE'
	#method = 'isomap'

	latent_d = dict()
	colors = {
		'base_conv': 'y',
		'base_resp': 'r',
		'bias_conv': 'k',
		'bias_nonc': 'b',
		}

	print('building data...')
	d_inp_enc = master.dataset.feed_data('test', max_n=n_sample, check_src=True, mix_ratio=(0.,1.))['inp_enc']
	latent_d['base_conv'] = master.model_encoder['S2S'].predict(d_inp_enc['ctxt'])
	if use_bias and 'AE' in master.prefix:
		latent_d['bias_nonc'] = master.model_encoder['AE'].predict(d_inp_enc['nonc'])
	#if use_bias and 'bias_conv' in master.dataset.files['test']:
	#	d_inp_enc = master.dataset.feed_data('test', max_n=n_sample, check_src=True, mix_ratio=(1.,0.))['inp_enc']
	#	latent_d['bias_conv'] = master.model_encoder['S2S'].predict(d_inp_enc['ctxt'])
	#else:
	d_inp_enc = master.dataset.feed_data('test', max_n=n_sample, check_src=True, mix_ratio=(0.,0.))['inp_enc']
	if 'AE' in master.prefix:
		#latent_d['base_nonc'] = master.model_encoder['AE'].predict(d_inp_enc['nonc'])
		latent_d['base_resp'] = master.model_encoder['AE'].predict(d_inp_enc['resp'])

	labels = list(sorted(latent_d.keys()))
	fname_suffix = args.restore.split('/')[-1].replace('.npz','')
	if use_bias:
		fname_suffix += '_wbias'
	n_labels = len(labels)
	latent = np.concatenate([latent_d[k] for k in labels], axis=0)
	print('latent.shape',latent.shape)

	print('plotting bit hist...')
	bins = np.linspace(-1,1,31)
	for k in latent_d:
		l = latent_d[k].ravel()
		freq, _, _ = plt.hist(l, bins=bins, color='w')
		plt.plot(bins[:-1], 100.*freq/sum(freq), colors[k]+'.-')
	plt.ylim([0,50])
	plt.savefig(fld_save+'/hist_%s.png'%fname_suffix)
	plt.close()
	
	print('plotting dist mat...')
	d_norm = np.sqrt(latent.shape[1])
	f, ax = plt.subplots()
	cax = ax.imshow(dist_mat(latent)/d_norm, cmap='bwr')
	#ax.set_title(model_name)
	f.colorbar(cax)

	ticks = []
	ticklabels = []
	n_prev = 0
	for i in range(n_labels):
		ticks.append(n_prev + n_sample/2)
		ticklabels.append(labels[i]+'\n')
		ticks.append(n_prev + n_sample)
		ticklabels.append('%i'%(n_sample * (i+1)))
		n_prev = n_prev + n_sample
	ax.set_xticks(ticks)
	ax.set_xticklabels(ticklabels)
	ax.xaxis.tick_top()
	ax.set_yticks(ticks)
	ax.set_yticklabels([s.strip('\n') for s in ticklabels])

	plt.savefig(fld_save+'/dist_%s.png'%fname_suffix)
	plt.close()

	if method == 'tSNE':
		approx = manifold.TSNE(init='pca', verbose=1).fit_transform(latent)
	elif method == 'MDS':
		approx = manifold.MDS(D, verbose=1, max_iter=500, n_init=1).fit_transform(latent)
	elif method == 'isomap':
		approx = manifold.Isomap().fit_transform(latent)
	else:
		raise ValueError

	f, ax = plt.subplots()
	for k in labels:
		ax.plot(np.nan, np.nan, colors[k]+'.', label=k)
	
	jj = list(range(approx.shape[0]))
	np.random.shuffle(jj)
	for j in jj:
		i_label = int(j/n_sample)
		ax.plot(approx[j, 0], approx[j, 1], colors[labels[i_label]]+'.')
		
	#plt.legend(loc='best')
	plt.title(model_name)
	#ax.set_xticks([])
	#ax.set_yticks([])
	plt.savefig(fld_save+'/%s_%s.png'%(method, fname_suffix))
	plt.show()






def cos_sim(a, b):
	#return 1. - scipy.spatial.distance.cosine(a, b)
	return np.inner(a, b)/np.linalg.norm(a)/np.linalg.norm(b)


def angel_hist(master, model_name, fld_save):
	from rand_decode import load_1toN_data
	data = load_1toN_data(master.dataset.generator['test'])
	angel = []
	n_sample = 1000

	extra_info = []

	for i in range(n_sample):
		if i%10 == 0:
			print(i)
		d = data[i]
		src_seq = np.reshape(d['src_seq'], [1,-1])
		latent_src = np.ravel(master.model_encoder['dial'].predict(src_seq))
		diff = []
		for ref_seq in d['ref_seqs']:
			ref_seq = np.reshape(ref_seq, [1,-1])
			latent_ref = np.ravel(master.model_encoder['auto'].predict(ref_seq))
			diff.append(latent_ref - latent_src)

		for i in range(len(diff) - 1):
			for j in range(i + 1, len(diff)):
				if str(d['ref_seqs'][i]) == str(d['ref_seqs'][j]):
					continue
				angel.append(cos_sim(diff[i], diff[j]))
				extra_info.append('%i\t%i'%(i, len(d['ref_seqs'])))

	with open(fld_save+'/angel.txt', 'w') as f:
		f.write('\n'.join([str(a) for a in angel]))
	with open(fld_save+'/angel_extra.txt', 'w') as f:
		f.write('\n'.join(extra_info))

	plt.hist(angel, bins=30)
	plt.title(model_name)
	plt.savefig(fld_save+'/angel.png')
	plt.show()




def plot_history(paths, labels, k, ix=-1, ax=None):
	if isinstance(paths, str):
		paths = [paths]
	import matplotlib.pyplot as plt

	def MA(y):
		window = 30
		ret = [np.nan] * len(y)
		for i in range(window, len(y)):
			ret[i] = np.mean(y[max(0, i - window + 1): i + 1])
		return ret

	def read_log(path, k):
		trained = np.nan
		xx = []
		yy = [[] for _ in range(4)]
		m = None
		for line in open(path):
			if line.startswith('***** trained '):
				trained = float(line.split(',')[0].split()[-2])
			if line.startswith(k):
				vv = [float(v) for v in line.replace(':','=').split('=')[-1].split(',')]
				if m is None:
					m = len(vv)
					print('expecting %i values'%m)
				else:
					if m!=len(vv):
						continue
				xx.append(trained)
				for i in range(len(vv)):
					yy[i].append(vv[i])
		return xx, yy[:m]

	if ax is None:
		_, ax = plt.subplots()

	color = ['r','b','k','m']
	if len(paths) > 0:
		for i, path in enumerate(paths):
			xx, yy = read_log(path, k)
			ss = path.split('/')
			label = ss[-1].replace('.txt','')
			ax.plot(xx, yy[ix], color=color[i], linestyle=':', alpha=0.5)
			ax.plot(xx, MA(yy[ix]), color=color[i], label=labels[i])
		ax.set_title(k + '[%i]'%ix)

	else:
		xx, yy = read_log(paths[0], k)
		for i in range(len(yy)):
			ax.plot(xx, yy[i], color=color[i], linestyle=':')
			ax.plot(xx, MA(yy[i]), color=color[i], label=str(i + 1))



def plot_multiple(kk, paths, labels):
	
	n_col = 4
	n_row = int(len(kk)/n_col)
	n_row = int(np.ceil(len(kk)/n_col))
	print('n_row = %i'%n_row)
	_, axs = plt.subplots(n_row, n_col, sharex=True)
	for i in range(len(kk)):
		k = kk[i]
		col = i%n_col
		row = int(i/n_col)
		ax = axs[row][col]
		if k.startswith('bleu') or k.startswith('corr'):
			ix = 2
		else:
			ix = -1
		plot_history(paths, labels, k, ix, ax=ax)
		#if i == 0:
		#	ax.legend(loc='best')
		ax.grid(True)
	plt.show()


if __name__ == '__main__':
	parser.add_argument('--vis_tp', default='clusters')
	parser.add_argument('--use_bias', type=int, default=1)
	parser.add_argument('--n_batch', type=int, default=5)
	args = parser.parse_args()
	print('>>>>> Not using GPU')
	os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
	
	master = run_master('vis', args)

	#if args.cpu_only:

	#fld = os.path.join(fld_model, model_name, 'vis')
	model_name = ''
	fld = master.fld + '/vis'
	print(fld)
	makedirs(fld)

	if args.vis_tp.startswith('interp'):
		if 'stry' in args.vis_tp:
			interp(master, model_name, fld, type_='stry')
		else:
			interp(master, model_name, fld, type_='resp')
	elif args.vis_tp == 'clusters':
		clusters(master, model_name, fld, use_bias=bool(args.use_bias), n_batch=args.n_batch)
	elif args.vis_tp == 'angel':
		angel_hist(master, model_name, fld)
	else:
		raise ValueError