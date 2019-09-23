"""
AUTHOR: 
Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""

import re, os, subprocess
from nltk.tokenize import TweetTokenizer

EQN_token = '_eqn_'
CITE_token = '_cite_'
IX_token = '_ix_'
MAX_UTT_LEN = 30	# maximum length of utterance allowed. if longer, ignore


def norm_sentence(txt):
	txt = txt.lower()

	# url and tag
	words = []
	for word in txt.lower().split():
		if word[0] == '#':	# don't allow tag
			continue
		i = word.find('http') 
		if i >= 0:
			word = word[:i] + ' ' + '__url__'
		words.append(word.strip())
	txt = ' '.join(words)

	# remove illegal char
	txt = txt.replace(chr(92),'')	# chr(92) = '\'. as twitter has 'b\/c' rather than 'b/c'
	txt = txt.replace("b/c","because").replace('j/k','just kidding').replace('w/o','without').replace('w/','with')
	txt = re.sub('__mention__','MENTION',txt)
	txt = re.sub('__url__','URL',txt)
	txt = re.sub(r"[^A-Za-z0-9():,.!?_'“” ]", " ", txt)
	txt = re.sub('MENTION','__mention__',txt)	
	txt = re.sub('URL','__url__',txt)	

	# contraction
	add_space = ["'s", "'m", "'re", "n't", "'ll","'ve","'d","'em"]
	tokenizer = TweetTokenizer(preserve_case=False)
	txt = ' ' + ' '.join(tokenizer.tokenize(txt)) + ' '
	txt = txt.replace(" won't ", " will n't ")
	txt = txt.replace(" can't ", " can n't ")
	for a in add_space:
		txt = txt.replace(a+' ', ' '+a+' ')
	
	# remove un-necessary space
	return ' '.join(txt.split())

def arxiv_del_bib(path):
	lines = []
	for line in open(path, encoding='utf-8', errors='ignore'):
		stop = False
		for c in ['begin{references}', 'begin{enumerate}', 'begin{thebibliography}']:
			if c in line:
				stop = True
		if stop:
			lines.append(chr(92) + 'end{document}\n\n')
			break
		else:
			lines.append(line.strip('\n'))
	with open(path+'.delbib', 'w', encoding='utf-8') as f:
		f.write('\n'.join(lines))


def arxiv_pandoc(fld):
	# preprocess arxiv latex file with pandoc
	# http://www.cs.cornell.edu/projects/kddcup/datasets.html
	# http://pandoc.org/index.html

	n = 0
	for fname in os.listdir(fld):
		if '.' in fname:
			continue

		path = fld + '/' + fname
		arxiv_del_bib(path)
		cmd = [
			'pandoc',
			'-f', 'latex',
			'-o', path + '_pandoc.txt',
			path + '.delbib'
			]
		process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
		process.communicate()
		print('='*10)
		print(path)

		n += 1
		#if n == 100:
		#	break
			

def arxiv_paragraph(path):
	def lines2paragraph(lines):
		p = ' '.join(lines)
		if len(p) == 0:
			return None
		if not p[0].isalpha():
			return None
		return arxiv_clean(p)
		
	paragraphs = []
	lines = []
	for line in open(path, encoding='utf-8', errors='ignore'):
		line = line.strip('\n').strip()
		if len(line) == 0:
			paragraph = lines2paragraph(lines)
			if paragraph is not None:
				paragraphs.append(paragraph)
			lines = []
		else:
			if len(re.sub(r"[^A-Za-z0-9]", "", line)) > 0:
				lines.append(line)

	paragraph = lines2paragraph(lines)
	if paragraph is not None:
		paragraphs.append(paragraph)

	with open(path+'.paragraph','w', encoding='utf-8') as f:
		f.write('\n'.join(paragraphs))



def arxiv_clean(p):
	# deal with equations and citations

	is_math = False
	i = 0
	s = ''
	while i < len(p):
		flag = False
		if p[i] == '$':
			flag = True
			if i + 1 < len(p) and p[i+1] == '$':
				i = i+1
		if flag:
			if not is_math:
				s += ' %s '%EQN_token
			is_math = not is_math
		elif not is_math:
			s += p[i]
		i += 1

	ww = []
	for w in s.split():
		if w == EQN_token:
			if len(ww) > 0 and ww[-1] == EQN_token:
				continue
			if len(ww) > 1 and ww[-1] == '.' and ww[-2] == EQN_token:
				continue
		if chr(92)+'[' in w or chr(92) + ']' in w or '[@' in w or w.startswith('@'):
			# citation
			w_ = CITE_token
			if w[-1]!=']':
				w_ += w[-1]
			w = w_
			if len(ww) > 0 and ww[-1].startswith(CITE_token):
				continue
		else:
			if w[0] == '(' and len(w) > 1 and w[1].isnumeric():
				w_ = IX_token
				if w[-1]!=')':
					w_ += w[-1]
				w = w_
		w = w.replace('[**','').replace('[*','').replace('**]','').replace('*]','')
		ww.append(w)

	return ' '.join(ww)


def arxiv_paragraph_all(fld):
	n = 0
	for fname in os.listdir(fld):
		if fname.endswith('_pandoc.txt'):
			path = fld + '/' + fname
			print(path)
			arxiv_paragraph(path)
			n += 1

def arxiv_utts(path):
	utts = []
	for p in open(path, encoding='utf-8'):
		p = p.strip('\n').replace(chr(92),'. ')
		for utt in p.split('. '):
			utt += '.'
			alpha = re.sub(r"[^a-z]", "", utt.lower())
			if len(alpha) < 5:
				continue
			utt = norm_sentence(utt)
			if len(utt.split()) > MAX_UTT_LEN:
				continue
			utts.append(utt)
	with open(path+'.utt', 'w', encoding='utf-8') as f:
		f.write('\n'.join(utts))
	return utts

def arxiv_utts_all(fld):
	utts = []
	n = 0
	for fname in os.listdir(fld):
		if fname.endswith('.paragraph'):
			path = fld + '/' + fname
			print(path)
			utts_ = arxiv_utts(path)
			for utt in utts_:
				if len(utt) > 10:
					utts.append(utt)
	utts = sorted(utts)
	with open(fld + '/../all.utt', 'w', encoding='utf-8') as f:
		f.write('\n'.join(utts))

def arxiv_filter(path):
	# make sure: 1) starts with some word, 2) longer than some min
	lines = []
	for line in open(path, encoding='utf-8'):
		ww = line.strip('\n').replace('“','"').replace('”','"').split()
		i0 = None
		for i in range(len(ww)):
			alpha = re.sub(r"[^a-z]", "", ww[i].lower())
			if len(alpha) > 1:
				i0 = i
				break
		if i0 is None:
			continue
		ww = ww[i0:]
		if len(re.sub(r"[^a-z]", "", ''.join(ww))) > 5:
			lines.append(' '.join(ww))
	with open(path+'.filtered', 'w', encoding='utf-8') as f:
		f.write('\n'.join(lines))


if __name__ == '__main__':
	years = range(1998, 2002+1)
	for year in years:
		fld = 'hep-th-%i/%i/'%(year, year)
		arxiv_pandoc(fld)
		arxiv_paragraph_all(fld)
		arxiv_utts_all(fld)
	for year in years:
		print(year)
		arxiv_filter('hep-th-%i/all.utt'%(year))
