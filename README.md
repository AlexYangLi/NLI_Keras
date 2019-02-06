### Natural Language Inference

Keras implementation (tensorflow backend) of natural language inference

### Models

- [InferSent, EMNLP2017](https://arxiv.org/pdf/1705.02364.pdf)

Conneau et al. Supervised Learning of Universal Sentence Representations from Natural Language Inference Data

### Environment
- python==3.6.4
- keras==2.2.4
- nltk==3.2.5
- sacred==0.7.4
- tensorflow=1.6.0

### Preparing Data

- NLI Data

download `SNIL`, `MultiNLI`,  `MedNLI` data, put them in `raw_data/` dir:  
    1. [SNIL](https://nlp.stanford.edu/projects/snli/)  
    2. [MultiNLI](http://www.nyu.edu/projects/bowman/multinli/)  
    3. [MedNLI](https://jgc128.github.io/mednli/)  

- Pre-trained embeddings

download pre-trained embeddings below, put them in `raw_data/word_embeddings` dir:  
    1. [glove_cc](http://nlp.stanford.edu/data/glove.840B.300d.zip)  
    2. [fasttext_wiki](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip), rename to `fasttext-wiki-news-300d-1M-subword.vec`  
    3. [fasttext_cc](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip), rename to `fasttext-crawl-300d-2M-subword.vec`  

### Pre-processing
```
python3 preprocess.py
```

### Training
```
python3 train.py
```

### Performance

- SNIL

| model                      | train(paper)| train | dev(paper) | dev    | test(paper) | test  |train_time(1 TITAN X)|
|----------------------------|-------------|-------|------------|--------|-------------|-------|---------------------|
|infersent(bilstm-max)       |   -	       | 90.54 |85.0        |85.43   | 84.5        |85.01  |01:28:26             |
|infersent(bilstm-mean)      |   -         |       |79.0        |        | 78.2        |       |                     |
|infersent(lstm)             |   -         |       |81.9        |        | 80.7        |       |                     |
|infersent(gru)              |   -         |       |82.4        |        | 81.8        |       |                     |
|indersent(bigru-last)       |   -         |       |81.3        |        | 80.9        |       |                     |
|infersent(bilstm-last)      |   -         |       |-           |        | -           |       |                     |
|infersent(inner-attention)  |   -         |       |82.3        |        | 82.5        |       |                     |
|infersent(hconv-net)        |   -         | 88.07 |83.7        |83.46   | 83.4        |83.23  |00:24:36             |


### Reference

- Part of my code are based on [mednil](https://github.com/jgc128/mednli). Great work, thanks!
