"Structuring Latent Spaces for Stylized Response Generation" at EMNLP'19, by Xiang Gao, Yizhe Zhang, Sungjin Lee, Michel Galley, Chris Brockett, Jianfeng Gao, Bill Dolan
# StyleFusion
[StyleFusion](https://arxiv.org/abs/1909.05361) jointly learns from a conversational dataset and other formats of text (e.g., non-parallel, non-conversational stylized text dataset). In our EMNLP 2019 [paper](https://github.com/golsun/StyleFusion/blob/master/EMNLP%20paper.pdf), we demonstrated its use to generate response in style of **Sherlock Holmes** and **arXiv**. StyleFusion is a generalized version of our previous work [SpaceFusion](https://github.com/golsun/SpaceFusion).

See an [introduction](https://mp.weixin.qq.com/s/rtAra15Qqnz9bLadSUSAlg) of our work (not official, by Shannon.AI, in Chinese)

## Dataset
In our paper, we trained the model using the following three datasets. 
* **Reddit**: the conversational dataset, can be generated using the [script](https://github.com/golsun/SpaceFusion/tree/master/data#multi-ref-reddit) hosted in our previous SpaceFusion work.
* **Sherlock Holmes**-style dataset is avaialble [here](https://github.com/golsun/StyleFusion/tree/master/data/Holmes)
* **arXiv**-style dataset can be obtained following instructions [here](https://github.com/golsun/StyleFusion/tree/master/data/arXiv)

## Usage
* to train a model `python src/main.py train`
* to interact with a trained model `python src/main.py cmd -restore=[path_to_model_file]`
* to interact with the [provided style classifiers](https://github.com/golsun/StyleFusion/tree/master/classifier) `python src/classifier.py [fld]`. `[fld]` is the folder where the classifier model exists, e.g., `classifier/Reddit_vs_arXiv/neural`

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


