# StyleFusion
StyleFusion is a generalized version of [SpaceFusion](https://github.com/golsun/SpaceFusion), which allows jointly learnining from a conversational dataset and other formats of text (e.g., non-parallel, non-conversational stylized text dataset). In out EMNLP 2019 paper, we demonstrated its use to generate stylized response.

## Usage
* to train the classifier `python src/classifier.py`
* to train a model `python src/main.py`
* to interact with a trained model `python src/main.py`

## Dataset
In our paper, we trained the model on a conversational dataset (Reddit) and then learn from a non-conversational stylized dataset (Holmes or arXiv). 
* the conversation Reddit dataset can be generated using this [script]()
* the processed Holmes data is avaialble [here]()
* the arXiv data can be downloaded from [here]() and then processed by this [script]()

## Citation
Please cite our EMNLP paper if this repo inspired your work :)
```
@article{gao2019stylefusion,
  title={Structuring Latent Spaces for Stylized Response Generation},
  author={Gao, Xiang and Zhang, Yizhe and Lee, Sungjin and Galley, Michel and Brockett, Chris and Gao, Jianfeng and Dolan, Bill},
  journal={EMNLP 2019},
  year={2019}
}
```

![](https://github.com/golsun/StyleFusion/blob/master/fig/intro_fig.PNG)


