from shared import *
from tf_lib import *
from dataset import *
from decode import *
from evaluate import *


"""
AUTHOR: Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""


class ModelBase:

	def __init__(self):
		self.fld = None				# str
		self.n_trained = None		# int
		self.max_n_trained = None	# int
		self.dataset = None			# Dataset obj
		self.extra = None			# list of str
		self.vali_data = None		# dict of list
		self.layers = None


	def init_log(self, new, args):
		
		# deal with existing fld
		if new and os.path.exists(self.fld):
			if PHILLY:
				suffix = 0
				while True:
					fld = self.fld + '_%i'%suffix
					if not os.path.exists(fld):
						self.fld = fld
						break
			else:
				if not PHILLY and not self.debug:
					print('%s\nalready exists, do you want to delete the folder? (y/n)'%self.fld)
					ans = input()
					if not ans.lower() == 'y':
						exit()
						
				print('deleting fld: '+self.fld)
				shutil.rmtree(self.fld)
				time.sleep(0.1)	
				print('fld deleted')

		self.log_train = self.fld  + '/train.txt'

		if new or PHILLY or hostname != 'MININT-3LHNLKS':
			makedirs(os.path.join(self.fld, 'models'))
			open(self.log_train, 'w')
			if not os.path.exists(self.fld + '/vocab.txt'):
				shutil.copyfile(self.dataset.path_vocab, self.fld + '/vocab.txt')

			ss = []
			for k in sorted(args.__dict__.keys()):
				ss.append('%s = %s'%(k, args.__dict__[k]))
			with open(self.fld + '/args.txt', 'w') as f:
				f.write('\n'.join(ss))

		if PHILLY:
			with open(self.log_train, 'a') as f:
				f.write('hostname:  %s\n'%hostname)
				f.write('data_path: %s\n'%DATA_PATH)
				f.write('out_path:  %s\n'%OUT_PATH)


	def train(self, batch_per_load=100):
		self.vali()
		while self.n_trained < self.max_n_trained:
			s = '\n***** trained %.3f M'%(self.n_trained/1e6)
			for tp in self.dataset.n_reset['train']:
				s += ', %s = %i'%(tp, self.dataset.n_reset['train'][tp])
			s += ' *****'
			write_log(self.log_train, s)
			self.train_a_load(batch_per_load)
			if self.debug:
				exit()


	def load_weights(self, path):
		self.prev_wt_fuse = None
		print('loading weights from %s'%path)
		npz = np.load(path, encoding='latin1')
		print(npz.files)
		weights = npz['layers'].item()
		for k in weights:
			s = ' '*(20-len(k)) + k + ': %i params: '%len(weights[k])
			for wt in weights[k]:
				s += str(wt.shape) + ', '
			print(s)
		for attr in self.extra:
			if attr in npz:
				if attr not in ['name']:
					setattr(self, attr, npz[attr])
			else:
				print('WARNING! attr %s not in npz'%attr)
		self.build_model(weights)
		self.build_model_test()


	def extract_weights(self):
		weights = dict()
		if self.layers is None:
			return weights
		for k in self.layers:
			weights[k] = self.layers[k].get_weights()
		return weights


	def save_weights(self):
		path = self.fld + '/models/%.1fM.npz'%(self.n_trained/1e6)
		weights = self.extract_weights()
		to_save = {'layers':weights}
		for attr in self.extra:
			to_save[attr] = getattr(self, attr)
		n_try = 0
		while n_try < 3:
			try:
				np.savez(path, **to_save)
				print('saved to: '+path)
				break
			except:
				n_try += 1
				print('cannot save, try %i'%n_try)
		return path

	def build_model_test(self):
		pass
	def build_model(self, weights=dict()):
		pass
	def train_a_load(self, batch_per_load):
		pass
	def set_extra(self, npz):
		pass




class Seq2SeqBase(ModelBase):

	def __init__(self, dataset, fld, args, new=False, allowed_words=None):

		self.dataset = dataset
		self.fld = fld
		self.allowed_words = allowed_words
		self.layers = None
		self.history = LossHistory()
		self.vali_data = None
		self.classifiers = []
		
		self.n_batch = 0
		self.prev_n_batch = 0
		self.dn_batch_vali = 100

		self.bias_conv = False # hasattr(self.dataset, 'files') and ('bias_conv' in self.dataset.files['train'])
		self.debug = args.debug
		self.token_embed_dim = args.token_embed_dim
		self.rnn_units = args.rnn_units
		self.encoder_depth = args.encoder_depth
		self.decoder_depth = args.decoder_depth
		self.lr = args.lr
		self.max_n_trained = args.max_n_trained
		self.randmix = False
		self.mix_ratio = (args.conv_mix_ratio, args.nonc_mix_ratio)
		
		if not self.bias_conv:
			assert(args.conv_mix_ratio == 0.)

		self.extra = ['name']
		self.init_extra(args)

		if hasattr(args, 'skip'):
			skip = int(1e6*args.skip)
		else:
			skip = 0
		self.dataset.skip(skip, self.mix_ratio, conv_only=(self.name=='s2s'))
		self.n_trained = skip
		self.init_log(new, args)
		self.build_model()
	

	def get_mix_ratio(self):
		if self.randmix:
			ret = []
			for ratio in self.mix_ratio:
				p = [1. - ratio, ratio]
				ret.append(np.random.choice([0.,1.], 1, p=p)[0])
			return tuple(ret)
		else:
			return self.mix_ratio


	def fit(self, inputs, outputs):
		n_try = 0
		if self.debug:
			self.model.fit(
					inputs, 
					outputs,
					batch_size=BATCH_SIZE,
					callbacks=[self.history],
					verbose=FIT_VERBOSE)
			return

		while n_try < 3:
			try:
				self.model.fit(
					inputs, 
					outputs,
					batch_size=BATCH_SIZE,
					callbacks=[self.history],
					verbose=FIT_VERBOSE)
				return
			except Exception as e:
				print('got error, sleeping')
				print('E'*20)
				print(e)
				print('E'*20)
				time.sleep(1)
				n_try += 1
		

	def _stacked_rnn(self, rnns, inputs, initial_states=None):
		if initial_states is None:
			initial_states = [None] * len(rnns)
		outputs, state = rnns[0](inputs, initial_state=initial_states[0])
		states = [state]
		for i in range(1, len(rnns)):
			outputs, state = rnns[i](outputs, initial_state=initial_states[i])
			states.append(state)
		return outputs, states


	def _build_encoder(self, inputs, prefix):
		_, encoder_states = self._stacked_rnn(
				[self.layers['%s_encoder_rnn_%i'%(prefix, i)] for i in range(self.encoder_depth)], 
				self.layers['embedding'](inputs))
		latent = encoder_states[-1]
		return latent


	def _build_decoder(self, input_seqs, input_states):
		"""
		for auto-regressive, states are returned and used as input for the generation of the next token
		for teacher-forcing, token already given, so only need init states
		"""

		decoder_outputs, decoder_states = self._stacked_rnn(
				[self.layers['decoder_rnn_%i'%i] for i in range(self.decoder_depth)], 
				self.layers['embedding'](input_seqs),
				input_states)
		decoder_outputs = self.layers['decoder_softmax'](decoder_outputs)
		return decoder_outputs, decoder_states


	def _create_layers(self, weights=dict()):
		layers = dict()

		name = 'embedding'
		params = _params(name, weights, {'mask_zero':True})
		layers[name] = Embedding(
				self.dataset.num_tokens + 1,		# +1 as mask_zero 
				self.token_embed_dim, 
				**params)

		for i in range(self.decoder_depth):
			name = 'decoder_rnn_%i'%i
			params = _params(name, weights, {'return_state':True, 'return_sequences':True})
			layers[name] = GRU(
				self.rnn_units, 
				**params)

		for prefix in self.prefix:
			for i in range(self.encoder_depth):
				name = '%s_encoder_rnn_%i'%(prefix, i)
				params = _params(name, weights, {'return_state':True, 'return_sequences':True})
				layers[name] = GRU(
						self.rnn_units, 
						**params)

		name = 'decoder_softmax'
		params = _params(name, weights, {'activation':'softmax'})
		layers[name] = Dense(
			self.dataset.num_tokens + 1, 		# +1 as mask_zero
			**params)

		return layers


	def build_model_test(self):
		
		#self.refresh_session()
		decoder_inputs = Input(shape=(None,), name='decoder_inputs')
		
		# encoder
		self.model_encoder = dict()
		self.model_tf = dict()
		self.tf_history = dict()
		for prefix in self.prefix:
			encoder_inputs = Input(shape=(None,), name=prefix+'_encoder_inputs')
			latent = self._build_encoder(encoder_inputs, prefix=prefix)
			self.model_encoder[prefix] = Model(encoder_inputs, latent)
			self.model_encoder[prefix]._make_predict_function()

			decoder_outputs, _ = self._build_decoder(decoder_inputs, [latent]*self.decoder_depth)
			self.model_tf[prefix] = Model([encoder_inputs, decoder_inputs], decoder_outputs)
			for layer in self.model_tf[prefix].layers:
				layer.trainable = False
			self.model_tf[prefix].compile(Adam(lr=0.), loss=_dec_loss)	# lr = 0 to use '.fit', which has callbacks, as '.evaluate'
			self.tf_history[prefix] = LossHistory()

		# decoder: autoregressive
		decoder_inital_states = []
		for i in range(self.decoder_depth):
			decoder_inital_states.append(Input(shape=(self.rnn_units,), name="decoder_inital_state_%i"%i))
		decoder_outputs, decoder_states = self._build_decoder(decoder_inputs, decoder_inital_states)
		model_decoder = Model(
				[decoder_inputs] + decoder_inital_states, 
				[decoder_outputs] + decoder_states)
		model_decoder._make_predict_function()
		self.decoder = Decoder(self.dataset, model_decoder, 
				self.decoder_depth, self.rnn_units, allowed_words=self.allowed_words)


	
	def get_vali_data(self):
		if self.vali_data is not None:
			#print('returning self.vali_data', self.vali_data)
			return self.vali_data
		print('getting vali data...')
		
		def _feed_vali(k):
			self.dataset.reset('vali')
			d = self.dataset.feed_data('vali', max_n=vali_size, check_src=True, mix_ratio=k, conv_only=(self.name=='s2s'))
			self.dataset.reset('vali')
			return d
		
		if self.debug:
			vali_size = BATCH_SIZE
		else:
			vali_size = 1000
		self.vali_data = _feed_vali((0, 1))
		"""
		self.vali_data['base'] = _feed_vali((0, 0))
		self.vali_data['mix'] = _feed_vali(self.mix_ratio)
		if self.bias_conv:
			self.vali_data['bias'] = _feed_vali((1, 1))
		else:
			self.vali_data['bias'] = _feed_vali((0, 1))
			"""
		return self.vali_data

	def vali(self):
		
		self.build_model_test()
		ss = []
		for inp in ['who is he ?', 'do you like this game ?', 'good morning .']:
			ss.append(infer_comb(inp, self))
		write_log(self.log_train, '\n'.join(ss))

		"""
		data = self.get_vali_data()
		if self.name.startswith('fuse'):
			r_rand = 0.1 * np.sqrt(self.rnn_units)
		else:
			r_rand = 0.
		#s_decoded = ''#eval_decoded(self, data, self.classifiers, r_rand=r_rand)[0]
		#s_surrogate = eval_surrogate(self, data)[0]
		#write_log(self.log_train, '\n' + s_decoded + '\n\n' + s_surrogate + '\n')
		"""
		self.prev_n_batch = self.n_batch

		# save --------------------
		self.save_weights()

	
	def init_extra(self, args):
		pass

	
	def train_a_load(self, batch_per_load):

		mix_ratio = self.get_mix_ratio()
		data = self.dataset.feed_data('train', BATCH_SIZE * batch_per_load, mix_ratio=mix_ratio, conv_only=(self.name == 's2s'))
		n_sample, inputs, outputs = self._inp_out_data(data)

		t0 = datetime.datetime.now()
		t0_str = str(t0).split('.')[0]
		write_log(self.log_train, 'start: %s'%t0_str + ', mix_ratio = '+str(mix_ratio))
		print('fitting...')

		self.fit(inputs, outputs)
		self.n_trained += n_sample
		self.n_batch += batch_per_load

		dt = (datetime.datetime.now() - t0).seconds
		loss = np.mean(self.history.losses)

		write_log(self.log_train, 'n_batch: %i, prev %i'%(self.n_batch, self.prev_n_batch))
		ss = ['spent: %i sec'%dt, 'train: %.4f'%loss]
		write_log(self.log_train, '\n'.join(ss))

		if not self.debug and (self.n_batch - self.prev_n_batch < self.dn_batch_vali):
			return

		# vali --------------------
		self.vali()

	def print_loss(self, loss_weights):
		s = 'loss: '+'-'*20 + '\n'
		for i in range(len(self.loss)):
			loss_name = str(self.loss[i])
			if loss_name.startswith('<func'):
				loss_name = loss_name.split()[1]
			s += '%6.2f '%loss_weights[i] + loss_name + '\n'
		s += '-'*20 + '\n'
		write_log(self.log_train, s)


class Seq2Seq(Seq2SeqBase):
	def init_extra(self, args):
		self.name = 's2s'
		self.prefix = ['S2S']

	def build_model(self, weights=dict()):
		self.layers = self._create_layers(weights)	# create new
		encoder_inputs = Input(shape=(None,), name='encoder_inputs')
		decoder_inputs = Input(shape=(None,), name='decoder_inputs')

		# connections: teacher forcing
		latent = self._build_encoder(encoder_inputs, self.prefix[0])
		decoder_outputs, _ = self._build_decoder(decoder_inputs, [latent]*self.decoder_depth)

		# models
		self.model = Model(
				[encoder_inputs, decoder_inputs], 	# [input sentences, ground-truth target sentences],
				decoder_outputs)					# shifted ground-truth sentences 
		self.model.compile(Adam(lr=self.lr), loss=_dec_loss)


	def _inp_out_data(self, data):
		inputs = [data['inp_enc']['ctxt'], data['inp_dec']['resp']]
		outputs = data['out_dec']['resp']
		return data['n_sample'], inputs, outputs

	


class VanillaMTask(Seq2SeqBase):

	def init_extra(self, args):
		self.name = 'mtask'
		self.loss = [
			_dec_loss,			# logP(resp | S2S), just the seq2seq loss
			_dec_loss, 			# logP(resp | AE_resp)
			_dec_loss, 			# logP(resp | AE_nonc)
			]
		self.prefix = ['AE','S2S']

	
	def build_model(self, weights=dict()):
		loss_weights = [1., 0.5, 0.5]
		self.layers = self._create_layers(weights)	# create new

		# inputs
		inp_enc_ctxt = Input(shape=(None,), name='inp_enc_ctxt')
		inp_enc_resp = Input(shape=(None,), name='inp_enc_resp')
		inp_dec_resp = Input(shape=(None,), name='inp_dec_resp')
		inp_enc_nonc = Input(shape=(None,), name='inp_enc_nonc')
		inp_dec_nonc = Input(shape=(None,), name='inp_dec_nonc')

		inps_enc = [inp_enc_ctxt, inp_enc_resp, inp_enc_nonc]
		inps_dec = [inp_dec_resp, inp_dec_nonc]
		inputs = inps_enc + inps_dec 
		
		# hiddens
		vec_s2s = self._build_encoder(inp_enc_ctxt, prefix='S2S')
		vec_ae_resp = self._build_encoder(inp_enc_resp, prefix='AE')
		vec_ae_nonc = self._build_encoder(inp_enc_nonc, prefix='AE')
	
		# outputs
		out_s2s, _     = self._build_decoder(inp_dec_resp, [vec_s2s]*self.decoder_depth)
		out_ae_resp, _ = self._build_decoder(inp_dec_nonc, [vec_ae_resp]*self.decoder_depth)
		out_ae_nonc, _ = self._build_decoder(inp_dec_nonc, [vec_ae_nonc]*self.decoder_depth)
		outputs = [out_s2s, out_ae_resp, out_ae_nonc]

		# compile
		self.print_loss(loss_weights)
		self.model = Model(inputs, outputs)
		self.model.compile(Adam(lr=self.lr), loss=self.loss, loss_weights=loss_weights)

	
	def _inp_out_data(self, data, u=None):
		n_sample = data['n_sample']
		if n_sample == 0:
			return n_sample, [], []

		inps_enc = [data['inp_enc']['ctxt'], data['inp_enc']['resp'], data['inp_enc']['nonc']]
		inps_dec = [data['inp_dec']['resp'], data['inp_dec']['nonc']]
		outs_dec = [data['out_dec']['resp'], data['out_dec']['resp'], data['out_dec']['nonc']]

		return n_sample, inps_enc + inps_dec, outs_dec



class StyleFusion(Seq2SeqBase):

	def init_extra(self, args):
		self.name = args.model_class.lower()
		assert(self.name in ['fuse','fuse1'])
		self.max_wt_dist = args.wt_dist
		self.stddev = args.stddev
		self.v1 = (self.name == 'fuse1')
		self.ablation = args.ablation
		
		if self.v1:
			# roughly, not exactly, follow SpaceFusion v1, as in https://arxiv.org/abs/1902.11205
			_dec_loss_ae = _dec_loss
			_dist_loss = _absdiff_dist_v1
		else:
			# v2, consider fuse with nonc
			_dec_loss_ae = _dec_loss_u		# interp(ae_resp, ae_nonc)
			if args.reld:
				_dist_loss = _relative_dist # consider all these terms d(s2s,resp), d(s2s,nonc), d(resp), d(nonc), d(s2s)
			else:
				_dist_loss = _absdiff_dist
			self.randmix = True				# binary batch mix

		self.loss = [
			_dec_loss,			# logP(resp | S2S), just the seq2seq loss
			_dec_loss,		    # logP(resp | interp), interp is between ctxt and resp, i.e. the 3rd term in Eq.3 in NAACL
			_dec_loss_ae,
			_dist_loss]

		self.prefix = ['AE','S2S']

	"""
	def refresh_session(self):
		K.clear_session()	# avoid building graph over and over to slow down everything
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		K.set_session(tf.Session(config=config))
		for clf in self.classifiers:
			clf.load()
			"""
	
	def build_model(self, weights=dict()):
		loss_weights = [1., 1., 1., 1.]
		if self.ablation:
			loss_weights = [1., 1., 0., 1.]	# disable L_{smooth,style}

		self.layers = self._create_layers(weights)	# create new

		noisy = Lambda(_add_noise, 
			arguments={'stddev':self.stddev}, 
			name='noisy')
		concat = Concatenate(name='concat_1', axis=-1)

		# inputs
		inp_enc_ctxt = Input(shape=(None,), name='inp_enc_ctxt')
		inp_enc_resp = Input(shape=(None,), name='inp_enc_resp')
		inp_dec_resp = Input(shape=(None,), name='inp_dec_resp')

		inps_enc = [inp_enc_ctxt, inp_enc_resp]
		inps_dec = [inp_dec_resp]

		inp_enc_nonc = Input(shape=(None,), name='inp_enc_nonc')
		inp_dec_nonc = Input(shape=(None,), name='inp_dec_nonc')
		inps_enc.append(inp_enc_nonc)
		inps_dec.append(inp_dec_nonc)
		inp_u = [Input(shape=(None,), name='inp_u')]		# rand drawn from U(0,1). each batch has the same value, see _inp_out_data

		inputs = inps_enc + inps_dec + inp_u		# match _inp_out_data

		# hiddens
		vec_s2s = self._build_encoder(inp_enc_ctxt, prefix='S2S')
		vec_ae_resp = self._build_encoder(inp_enc_resp, prefix='AE')
		vec_ae_nonc = self._build_encoder(inp_enc_nonc, prefix='AE')
		vec_interp_resp = noisy(Lambda(_interp, name='interp_resp')([vec_s2s, vec_ae_resp] + inp_u))
		
		# outputs
		out_s2s, _ = self._build_decoder(inp_dec_resp, [vec_s2s]*self.decoder_depth)
		out_interp_resp, _ = self._build_decoder(inp_dec_resp, [vec_interp_resp]*self.decoder_depth)
		if self.v1:
			out_ae, _ = self._build_decoder(inp_dec_nonc, [vec_ae_nonc]*self.decoder_depth)
		else:
			vec_interp_ae = noisy(Lambda(_interp, name='interp_ae')([vec_ae_resp, vec_ae_nonc] + inp_u))
			out_interp_ae_resp, _ = self._build_decoder(inp_dec_resp, [vec_interp_ae]*self.decoder_depth)
			out_interp_ae_nonc, _ = self._build_decoder(inp_dec_nonc, [vec_interp_ae]*self.decoder_depth)
			out_ae = concat([out_interp_ae_resp, out_interp_ae_nonc])
		outs_dec = [out_s2s, out_interp_resp, out_ae]
		outs_dist = concat([vec_s2s, vec_ae_resp, vec_ae_nonc])
		outputs = outs_dec + [outs_dist]

		# compile
		self.print_loss(loss_weights)
		self.model = Model(inputs, outputs)
		self.model.compile(Adam(lr=self.lr), loss=self.loss, loss_weights=loss_weights)

	
	def _inp_out_data(self, data, u=None):
		n_sample = data['n_sample']
		if n_sample == 0:
			return n_sample, [], []
		if u is None:
			u = np.random.random(n_sample)
		else:
			u = np.array([u] * n_sample)

		inps_enc = [data['inp_enc']['ctxt'], data['inp_enc']['resp']]
		inps_dec = [data['inp_dec']['resp']]
		outs_dec = [data['out_dec']['resp'], data['out_dec']['resp']]

		inps_enc.append(data['inp_enc']['nonc'])
		inps_dec.append(data['inp_dec']['nonc'])
		inputs = inps_enc + inps_dec + [u]


		if self.v1:
			outs_dec.append(data['out_dec']['nonc'])
		else:
			_, l, v = data['out_dec']['resp'].shape
			out_interp_nonc = np.zeros([n_sample, l, v*2+1])	
			out_interp_nonc[:,:,:v] = data['out_dec']['resp']
			out_interp_nonc[:,:,v:v*2] = data['out_dec']['nonc']
			for t in range(l):
				out_interp_nonc[:,t,-1] = u
			outs_dec.append(out_interp_nonc)

		outputs = outs_dec + [np.zeros((n_sample, 1))]
		return n_sample, inputs, outputs




class LossHistory(Callback):
	def reset(self):
		self.losses = []

	def on_train_begin(self, logs={}):
		self.reset()

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))


def _params(name, weights, extra=dict()):
	params = {'name':name}
	if name in weights:
		params['weights'] = weights[name]
	for k in extra:
		params[k] = extra[k]
	return params


def write_log(path, s, PRINT=True, mode='a'):
	if PRINT:
		print(s)
		sys.stdout.flush()
	if not s.endswith('\n'):
		s += '\n'

	if PHILLY:
		n_try = 0
		while n_try < 3:
			try:
				with open(path, mode) as f:
					f.write(s)
				break
			except:# PermissionError as e:
				#print(e)
				print('cannot write_log, sleeping...')
				time.sleep(2)
				n_try += 1
	else:
		with open(path, mode) as f:
			f.write(s)



# ------------------- customized loss --------------------

def _dist_1nn(a, b=None):
	n = BATCH_SIZE
	expanded_a = tf.expand_dims(a, 1)
	if b is None:
		b = a
	expanded_b = tf.expand_dims(b, 0)
	d_squared = tf.reduce_mean(tf.squared_difference(expanded_a, expanded_b), 2)
	mat = tf.sqrt(tf.maximum(0., d_squared))

	wt = 1./(mat + tf.eye(n) * 1000 + 1e-6)
	sum_wt = tf.reshape(tf.reduce_sum(wt, axis=1), [n, 1])
	sum_wt = tf.tile(sum_wt, [1,n])
	wt = wt/sum_wt

	d1nn = tf.reduce_sum(mat * wt, axis=1)
	d1nn = tf.reduce_mean(d1nn)
	return d1nn


def _cross_inner(vecs, v1=False):
	def sqrt_mse(a, b=None, shuffle=True, cap=None):
		if b is None:
			b = a
		if shuffle:
			#diff = a - tf.random_shuffle(b)
			_, d = a.shape
			n = BATCH_SIZE - 1
			diff = tf.slice(a, [1,0], [n,d]) - tf.slice(b, [0,0], [n,d])
		else:
			diff = a - b
		squared = tf.pow(diff, 2)
		if cap is not None:
			squared = tf.minimum(cap**2, squared)
		return tf.sqrt(tf.reduce_mean(squared))

	vec_s2s, vec_ae_resp, vec_ae_nonc = tf.split(vecs, 3, axis=-1)
	cross_resp = sqrt_mse(vec_s2s, vec_ae_resp, shuffle=False)
	inner_s2s_resp = _dist_1nn(vec_s2s)
	inner_ae_nonc = _dist_1nn(vec_ae_nonc)

	if v1:
		print('*'*10 + ' [WARNING] Using v1 cross_inner ' + '*'*10)
		return cross_resp, inner_s2s_resp + inner_ae_nonc
	else:
		cross_s2s_nonc = _dist_1nn(vec_s2s, vec_ae_nonc)
		inner_ae_resp = _dist_1nn(vec_ae_resp)
		cross = 0.5 * (cross_resp + cross_s2s_nonc)
		inner = tf.minimum(tf.minimum(inner_s2s_resp, inner_ae_resp), inner_ae_nonc)
		return cross, inner


def _relative_dist(_, y_pred): 
	cross, inner = _cross_inner(y_pred)
	return cross / inner

def _absdiff_dist(_, y_pred):
	cross, inner = _cross_inner(y_pred)
	return cross - inner

def _absdiff_dist_v1(_, y_pred):
	cross, inner = _cross_inner(y_pred, v1=True)
	return cross - inner


def _dec_loss(y_true, y_pred):
	# to compute - logP(resp|vec_interp_resp)
	return tf.reduce_mean(keras.losses.categorical_crossentropy(y_true, y_pred))


def _dec_loss_u(y_true, y_pred):
	# to compute u * logP(resp|vec_interp_ae) + (1-u) * logP(nonc|vec_interp_ae)
	# where vec_interp_ae = u * vec_resp_ae + (1-u) * vec_nonc_ae
	# y_true = concat([y_resp, y_nonc, u]), shape = [BATCH_SIZE, seq_len, 2 * vocab_size + 1], see out_interp_nonc in _in_out_data
	# y_pred = concat([y_resp_pred, y_nonc_pred])

	y_resp_pred, y_nonc_pred = tf.split(y_pred, 2, axis=-1)
	vocab_size = tf.cast(y_resp_pred.shape[2], tf.int32)
	y_resp, y_nonc, u = tf.split(y_true, [vocab_size, vocab_size, 1], axis=-1)
	u = u[:,:,0]		# like tf.squeeze, so [BATCH_SIZE, seq_len]
	loss_resp = keras.losses.categorical_crossentropy(y_resp, y_resp_pred)	# [BATCH_SIZE, seq_len]
	loss_nonc = keras.losses.categorical_crossentropy(y_nonc, y_nonc_pred)

	loss = u * loss_resp + (1. - u) * loss_nonc	# [BATCH_SIZE, seq_len]
	return tf.reduce_mean(loss)

# ------------------- customized layers --------------------

def _add_noise(mu, stddev):
	eps = K.random_normal(shape=K.shape(mu))
	return mu + tf.multiply(eps, stddev)

def _interp(inp):
	if len(inp) == 2:
		a, b = inp
		u = K.random_uniform(shape=(K.shape(a)[0], 1))
	else:
		a, b, u = inp
	u = K.tile(K.reshape(u, [-1,1]), [1, K.shape(a)[1]])	# repeat along axis=1
	#return a + tf.multiply(b - a, u)
	return tf.multiply(a, u) + tf.multiply(b, 1 - u)


def convert_model_vocab(path_npz_old, path_npz_new, path_vocab_old, path_vocab_new):
	if os.path.exists(path_npz_new):
		print('already exists: '+path_npz_new)
		return
	_, token2index_old = load_vocab(path_vocab_old)
	index2token_new, _ = load_vocab(path_vocab_new)
	n_old = max(token2index_old.values()) + 1
	n_new = max(index2token_new.keys()) + 1
	print('vocab: %i => %i'%(n_old, n_new))

	new2old = dict()
	ix_unk_old = token2index_old[UNK_token]
	for ix in index2token_new:
		token = index2token_new[ix]
		new2old[ix] = token2index_old.get(token, ix_unk_old)

	print('loading from: '+str(path_npz_old))
	npz = np.load(path_npz_old, encoding='latin1')
	weights = npz['layers'].item()
	embedding_old = weights['embedding'][0]
	softmax_wt_old = weights['decoder_softmax'][0]
	softmax_bias_old = weights['decoder_softmax'][1]

	n_old_loaded, dim = embedding_old.shape
	assert(n_old_loaded == n_old)
	embedding_new = np.zeros((n_new, dim))
	softmax_wt_new = np.zeros((dim, n_new))
	softmax_bias_new = np.zeros((n_new,))

	print('   embedding: ' + str(embedding_old.shape) + ' => ' + str(embedding_new.shape))
	print('  softmax_wt: ' + str(softmax_wt_old.shape) + ' => ' + str(softmax_wt_new.shape))
	print('softmax_bias: ' + str(softmax_bias_old.shape) + ' => ' + str(softmax_bias_new.shape))

	# PAD
	embedding_new[0,:] = embedding_old[0, :]	
	softmax_wt_new[:, 0] = softmax_wt_old[:, 0]
	softmax_bias_new[0] = softmax_bias_old[0]

	for ix in index2token_new:
		embedding_new[ix, :] = embedding_old[new2old[ix], :]
		softmax_wt_new[:, ix] = softmax_wt_old[:, new2old[ix]]
		softmax_bias_new[ix] = softmax_bias_old[new2old[ix]]

	weights['embedding'] = [embedding_new]
	weights['decoder_softmax'] = [softmax_wt_new, softmax_bias_new]

	print('saving to: '+str(path_npz_new))
	to_save = {'layers':weights}
	for k in npz.files:
		if k != 'layers' and 'mix' not in k:
			to_save[k] = npz[k]
	np.savez(path_npz_new, **to_save)
