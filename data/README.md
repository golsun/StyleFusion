
## Format
The model needs base conversational corpus (`base`, e.g. **Reddit**) as well as a stylized corpus (`bias`, e.g. **arXiv**).
The `base` corpus should be conversational (so `base_conv`), and the `bias` corpus doesn't have to be. so we only need `bias_nonc` (`nonc` means non-conversational)
To sum up, at least these files are required: `base_conv_XXX.num`, `bias_nonc_XXX.num` and `vocab.txt`, where `XXX` is `train`, `vali`, or `test`.
See more discusion [here](https://github.com/golsun/StyleFusion/issues/3)


* `vocab.txt` is the vocab list of tokens. 
  * The first three token must be `_SOS_`, `_EOS_` and `_UNK_`, which represent "start of sentence", "end of sentence", and "unknown token".
  * The line ID (starts from 1, 0 is reserved for padding) of `vocab.txt` is the token index used in `*.num` files. For examples, unknown tokens will be represented by `3` which is the token index of `_UNK_`. 

* `*.num` files are sentences (in form of seq of token index), 
  * for `conv`, each line is `src \t tgt`, where `\t` is the tab delimiter
  * for `nonc`, each line is a sentence.

You may build a vocab using the [build_vocab](https://github.com/golsun/NLP-tools/blob/master/data_prepare.py#L266) function to generate `vocab.txt`,
and then convert a raw text files to `*.num` 
(e.g. [train.txt](https://github.com/golsun/SpaceFusion/blob/master/data/toy/train.txt) to [train.num](https://github.com/golsun/SpaceFusion/blob/master/data/toy/train.num))
by the [text2num](https://github.com/golsun/NLP-tools/blob/master/data_prepare.py#L381) function


## Dataset
In our paper, we trained the model using the following three datasets. 
* **Reddit**: the conversational dataset (`base_conv`), can be generated using this [script](https://github.com/golsun/SpaceFusion/tree/master/data#multi-ref-reddit).
* **Sherlock Holmes**, one of style dataset (`bias_nonc`), avaialble [here](https://github.com/golsun/StyleFusion/tree/master/data/Holmes)
* **arXiv**, another style corpus (`bias_nonc`), can be obtained following instructions [here](https://github.com/golsun/StyleFusion/tree/master/data/arXiv)
* A [toy dataset](https://github.com/golsun/StyleFusion/tree/master/data/toy) is provied as an example following the format described above.

