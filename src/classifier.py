from shared import *
from tf_lib import *
import json
from dataset import load_vocab
from sklearn import linear_model
import pickle

"""
AUTHOR: 
Sean Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""

class ClassifierNeural():

	def __init__(self, fld):
		params = json.load(open(fld + '/args.json'))
		if params['tgt_only']:
			self.prefix = ['tgt']
		else:
			self.prefix = ['src','tgt']
		self.encoder_depth = params['encoder_depth']
		self.rnn_units = params['rnn_units']
		self.mlp_depth = params['mlp_depth']
		self.mlp_units = params['mlp_units']
		self.include_punc = params['include_punc']
		self.index2token, self.token2index = load_vocab(fld + '/vocab.txt')

		self.fld = fld
		self.load()


	def load(self):
		self.build_model()
		self.model.load_weights(self.fld+'/model.h5')


	def _create_layers(self):
		layers = dict()

		layers['embedding'] = Embedding(
				max(self.index2token.keys()) + 1,		# +1 as mask_zero 
				self.rnn_units, mask_zero=True, 
				name='embedding')

		for prefix in self.prefix:
			for i in range(self.encoder_depth):
				name = '%s_encoder_rnn_%i'%(prefix, i)
				layers[name] = GRU(
						self.rnn_units, 
						return_state=True,
						return_sequences=True, 
						name=name)

		for i in range(self.mlp_depth - 1):
			name = 'mlp_%i'%i
			layers[name] = Dense(
				self.mlp_units, 
				activation='tanh', name=name)

		name = 'mlp_%i'%(self.mlp_depth - 1)
		layers[name] = Dense(1, activation='sigmoid', name=name)
		return layers


	def _stacked_rnn(self, rnns, inputs, initial_states=None):
		if initial_states is None:
			initial_states = [None] * len(rnns)

		outputs, state = rnns[0](inputs, initial_state=initial_states[0])
		states = [state]
		for i in range(1, len(rnns)):
			outputs, state = rnns[i](outputs, initial_state=initial_states[i])
			states.append(state)
		return outputs, states


	def _build_encoder(self, inputs, layers, prefix):
		_, encoder_states = self._stacked_rnn(
				[layers['%s_encoder_rnn_%i'%(prefix, i)] for i in range(self.encoder_depth)], 
				layers['embedding'](inputs))
		latent = encoder_states[-1]
		return latent


	def build_model(self):
		layers = self._create_layers()

		encoder_inputs = dict()
		latents = []
		for prefix in self.prefix:
			encoder_inputs[prefix] = Input(shape=(None,), name=prefix+'_encoder_inputs')
			latents.append(self._build_encoder(encoder_inputs[prefix], layers, prefix=prefix))

		if len(self.prefix) > 1:
			out = Concatenate()(latents)
			inp = [encoder_inputs['src'], encoder_inputs['tgt']]
		else:
			out = latents[0]
			inp = encoder_inputs[self.prefix[0]]
		for i in range(self.mlp_depth):
			out = layers['mlp_%i'%i](out)

		self.model = Model(inp, out)
		self.model.compile(optimizer=Adam(lr=0), loss='binary_crossentropy')


	def txt2seq(self, txt):
		tokens = txt.strip().split(' ')
		seq = []
		ix_unk = self.token2index[UNK_token]
		for token in tokens:
			if self.include_punc or is_word(token):		# skip punctuation if necessary
				seq.append(self.token2index.get(token, ix_unk))
		return seq

	def seq2txt(self, seq):
		return ' '.join([self.index2token[i] for i in seq])

	def txts2mat(self, txts, max_len=30):
		if isinstance(txts, str):
			txts = [txts]
		data = np.zeros((len(txts), max_len))
		for j, txt in enumerate(txts):
			seq = self.txt2seq(txt.strip(EOS_token).strip())	# stripped EOS_token here
			for t in range(min(max_len, len(seq))):
				data[j, t] = seq[t]
		return data

	def predict(self, txts):
		mat = self.txts2mat(txts)
		return self.model.predict(mat).ravel()




class ClassifierNgram:

    def __init__(self, fld, ngram, include_punc=False):
        self.fld = fld
        self.ngram2ix = dict()
        self.ngram = ngram
        self.include_punc = include_punc

        fname = '%igram'%ngram
        if include_punc:
            fname +=  '.include_punc'
        self.path_prefix = fld + '/' + fname
        for i, line in enumerate(open(self.path_prefix + '.txt', encoding='utf-8')):
            ngram = line.strip('\n')
            self.ngram2ix[ngram] = i
            assert(self.ngram == len(ngram.split()))
        self.vocab_size = i + 1
        print('loaded %i %igram'%(self.vocab_size, self.ngram))
        #self.model = LogisticRegression(solver='sag')#, max_iter=10)
        self.model = linear_model.SGDClassifier(loss='log', random_state=9, max_iter=1, tol=1e-3)

    def txts2mat(self, txts):
        X = np.zeros((len(txts), self.vocab_size))
        for i, txt in enumerate(txts):
            ww = txt2ww(txt, self.include_punc)
            for t in range(self.ngram, len(ww) + 1):
                ngram = ' '.join(ww[t - self.ngram: t])
                j = self.ngram2ix.get(ngram, None)
                if j is not None:
                    X[i, j] = 1.
        return X

    def load(self):
        self.model = pickle.load(open(self.path_prefix + '.p', 'rb'))

    def predict(self, txts):
        data = self.txts2mat(txts)
        prob = self.model.predict_proba(data)
        return prob[:,1]




class ClassifierNgramEnsemble:
                
    def __init__(self, fld, include_punc=False, max_ngram=4):
        self.fld = fld
        self.children = dict()
        self.wt = dict()
        for ngram in range(1, max_ngram + 1):
            self.children[ngram] = ClassifierNgram(fld, ngram, include_punc)
            self.children[ngram].load()
            acc = float(open(self.children[ngram].path_prefix + '.acc').readline().strip('\n'))
            self.wt[ngram] = 2. * max(0, acc - 0.5)

    def predict(self, txts):
        avg_scores = np.array([0.] * len(txts))
        for ngram in self.children:
            scores = self.children[ngram].predict(txts)
            avg_scores += scores * self.wt[ngram]
        return avg_scores / sum(self.wt.values())


def is_word(token):
	for c in token:
		if c.isalpha():
			return True
	return False




def load_classifier(fld, args=None):
	if fld.endswith('ngram'):
		return ClassifierNgramEnsemble(fld)
	elif fld.endswith('neural'):
		return ClassifierNeural(fld)
	else:
		raise ValueError
	

def clf_interact(fld):
	clf = load_classifier(fld)
	while True:
		print('\n---- please input ----')
		txt = input()
		if txt == '':
			break
		score = clf.predict([txt])[0]
		print('%.4f'%score)


def clf_eval(path):
    # path is a tsv, last col is hyp
    clf = load_classifier(fld)
    sum_score = 0
    n = 0
    for line in open(path, encoding='utf-8'):
        txt = line.strip('\n').split('\t')[-1].lower()
        sum_score += clf.predict([txt])[0]
        n += 1
        if n % 100 == 0:
            print('eval %i lines'%n)
    print('finally %i samples'%n)
    print('avg style score: %.4f'%(sum_score/n))


def txt2ww(txt, include_punc):
    ww = [SOS_token]
    for w in txt.split():
        if include_punc or is_word(w):
            ww.append(w)
    ww.append(EOS_token)
    return ww


def score_file(path, name, col=1):
	clf = load_classifier(name)
	txts = []
	for line in open(path, encoding='utf-8'):
		txts.append(line.strip('\n').split('\t')[col])
		if len(txts) == 1500:
			break
	print('scoring...')
	print(np.mean(clf.predict(txts)))


class Classifier1gramCount:
    def __init__(self, fld):
        self.fld = fld

    def fit(self, min_freq=60, max_n=1e5):
        scores = dict()
        n = 0
        for line in open(self.fld + '/all.txt', encoding='utf-8'):
            n += 1
            cells = line.strip('\n').split('\t')
            if len(cells) != 2:
                print(cells)
                exit()
            txt, score = cells
            for w in set(txt.strip().split()):
                if is_word(w):
                    if w not in scores:
                        scores[w] = []
                    scores[w].append(float(score))
            if n == max_n:
                break


        lines = ['\t'.join(['word', 'avg', 'se', 'count'])]
        for w in scores:
            count = len(scores[w])
            if count < min_freq:
                continue
            avg = np.mean(scores[w])
            se = np.std(scores[w])/np.sqrt(count)
            lines.append('\t'.join([w, '%.4f'%avg, '%.4f'%se, '%i'%count]))

        with open(self.fld + '/count.tsv', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def load(self):
        self.coef = dict()
        f = open(self.fld + '/count.tsv', encoding='utf-8')
        header = f.readline()
        for line in f:
            w, avg = line.strip('\n').split('\t')[:2]
            self.coef[w] = float(avg)

    def corpus_score(self, txts, kw=100):
        scores = []
        coef_w = []
        for w in self.coef:
            coef_w.append((self.coef[w], w))
        coef_w = sorted(coef_w, reverse=True)[:kw]
        print('last:',coef_w[-1])
        keywords = set([w for _, w in coef_w])

        #total_joint = 0
        #total = 0

        for txt in txts:
            words = set()
            for w in txt.strip().split():
                if is_word(w):
                    words.add(w)
            joint = words & keywords
            scores.append(len(joint)/len(words))
            #total_joint += len(joint)
            #total += len(words)
        return np.mean(scores), np.std(scores)/np.sqrt(len(scores))
        #return total_joint/total


    def test(self, kw=100):
        import matplotlib.pyplot as plt

        txts = []
        labels = []
        for line in open(self.fld + '/sorted_avg.tsv', encoding='utf-8'):
            txt, label = line.strip('\n').split('\t')
            txts.append(txt)
            labels.append(float(label))

        i0 = 0
        human = []
        pred = []
        while True:
            i1 = i0 + 100
            if i1 >= len(txts):
                break
            human.append(np.mean(labels[i0:i1]))
            pred.append(self.corpus_score(txts[i0:i1], kw=kw))
            i0 = i1

        plt.plot(human, pred, '.')
        plt.xlabel('human')
        plt.xlabel('metric (ratio of keywords)')
        plt.title('corr = %.4f'%np.corrcoef(human, pred)[0][1])
        plt.savefig(self.fld + '/test_corr_kw%i.png'%kw)


if __name__ == '__main__':
    # e.g. `python src/classifier.py classifier/Reddit_vs_arXiv/neural' for interaction
    # e.g. `python src/classifier.py classifier/Reddit_vs_arXiv/neural path/to/hyp/file.tsv' for evaluating a file
    fld_model = sys.argv[1]  # e.g.
    if len(sys.argv) == 2:
        clf_interact(fld_model)
    elif len(sys.argv) == 3:
        path_hyp = sys.argv[2]
        clf_eval(fld_model, path_hyp)