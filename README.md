# StyleFusion
code/data for EMNLP'19 paper [Structuring Latent Spaces for Stylized Response Generation](https://arxiv.org/abs/1909.05361).

Designed to build a **stylized** dialogue response generator, StyleFusion jointly learns from a conversational dataset and other formats of text (e.g., non-parallel, non-conversational stylized text dataset). In our EMNLP 2019 paper, we demonstrated its use to generate response in style of **Sherlock Holmes** and **arXiv**. StyleFusion is a generalized version of our previous work [SpaceFusion](https://github.com/golsun/SpaceFusion).

More documents:
* our EMNLP'19 [paper](https://arxiv.org/abs/1909.05361) and [poster](https://github.com/golsun/StyleFusion/blob/master/EMNLP%20poster.pdf).
* A nice [introduction](https://mp.weixin.qq.com/s/rtAra15Qqnz9bLadSUSAlg) of our work (not official, by Shannon.AI, in Chinese)

## Dataset
In our paper, we trained the model using the following three datasets. 
* **Reddit**: the conversational dataset (`base_conv`), can be generated using this [script](https://github.com/golsun/SpaceFusion/tree/master/data#multi-ref-reddit).
* **Sherlock Holmes**, one of style dataset (`bias_nonc`), avaialble [here](https://github.com/golsun/StyleFusion/tree/master/data/Holmes)
* **arXiv**, another style corpus (`bias_nonc`), can be obtained following instructions [here](https://github.com/golsun/StyleFusion/tree/master/data/arXiv)
* A [toy dataset](https://github.com/golsun/StyleFusion/tree/master/data/toy) is provied as an example following the format described above.
* We also provided the **test data** [here](https://github.com/golsun/StyleFusion/tree/master/data/test)

See [here](https://github.com/golsun/StyleFusion/blob/master/data/README.md) for more details and instructions.

## Usage
* to train a model `python src/main.py train`
* to interact with a trained model `python src/main.py cmd --restore=[path_to_model_file]`
* using the [provided style classifiers](https://github.com/golsun/StyleFusion/tree/master/classifier) 
   * interactive demo: `python src/classifier.py [fld_clf]`, where `[fld_clf]` is the folder where the classifier model exists, e.g., `classifier/Reddit_vs_arXiv/neural`
   * evaluate a tsv file: `python src/classifier.py [fld_clf] [path_to_be_evaluated]`.

## Citation
Please cite our EMNLP paper if this repo is useful to your work :)
```
@article{gao2019stylefusion,
  title={Structuring Latent Spaces for Stylized Response Generation},
  author={Gao, Xiang and Zhang, Yizhe and Lee, Sungjin and Galley, Michel and Brockett, Chris and Gao, Jianfeng and Dolan, Bill},
  journal={EMNLP 2019},
  year={2019}
}
```

![](https://github.com/golsun/StyleFusion/blob/master/fig/intro_fig.png)


