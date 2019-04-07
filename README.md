### Natural Language Inference

Keras implementation (tensorflow backend) of natural language inference

### Models

- [Decomposable Attention, EMNLP2016](https://arxiv.org/pdf/1606.01933v1.pdf)  
Parikh et al. A Decomposable Attention Model for Natural Language Inference.

- [InferSent, EMNLP2017](https://arxiv.org/pdf/1705.02364.pdf)  
Conneau et al. Supervised Learning of Universal Sentence Representations from Natural Language Inference Data.

- [ESIM, ACL2017](https://arxiv.org/pdf/1609.06038.pdf)  
Chen rt al. Enhanced LSTM for Natural Language Inference.

- Siamese Bilstm  
using a bilstm based siamese architecture(two networks with the same structure and the same weight, each process one sentence in a pair) to model both premise and hypothesis.

- Siamese CNN    
using a text-cnn based siamese architecture to model both premise and hypothesis.

- Siamese LSTMCNN  

- IA-CNN    
apply interaction attention (same as in esim model) to premise and hypothesis before feeding them into CNN network.

### Optimization Method

- [Cyclical Learning Rate](https://arxiv.org/pdf/1506.01186.pdf)  
cycles the learning rate between two boundaries with some constant frequency

- [SWA](https://arxiv.org/pdf/1803.05407.pdf)  
apply weight averageing at the end of each epoch of training to improve generalization

### Environment
- python==3.6.4
- keras==2.2.4
- nltk==3.2.5
- tensorflow=1.8.0
- fasttext
- glove_python
- tensorflow_hub


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
    4. [tfhub_elmo_2](https://tfhub.dev/google/elmo/2?tf-hub-format=compressed), untar the file, put all the files in a folder named `tfhub_elmo_2`
    5. original_elmo_5.5B: [options file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json)
        [weights file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5)
    6. [tf_bert](https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1?tf-hub-format=compressed)

### Pre-processing
```
python3 preprocess.py
python3 prepare_features.py
```

### Training
```
python3 train.py
```

### Performance (Accuracy)

- SNIL

| model                      | batch_size | optimizer  | embedding   | train(paper)| train | dev(paper) | dev    | test(paper) | test  |train_time(1 TITAN X)|
|----------------------------|------------|------------|-------------|-------------|-------|------------|--------|-------------|-------|---------------------|
|decomposable                |   512      |   adam     |glove_cc_fix |   89.5      | 87.52 |-           |81.52   | 86.3        |81.19  |00:12:53             |
|decomposable(intra-sentence)|            |            |             |   90.5      |       |-           |        | 86.8        |       |                     |
|infersent(bilstm-max)       |   64       |   adam     |glove_cc_fix |   -	       | 90.54 |85.0        |85.43   | 84.5        |85.01  |01:28:26             |
|infersent(bilstm-mean)      |   64       |   adam     |glove_cc_fix |   -         | 87.33 |79.0        |83.62   | 78.2        |83.62  |01:14:32             |
|infersent(lstm)             |   64       |   adam     |glove_cc_fix |   -         | 92.09 |81.9        |84.20   | 80.7        |83.19  |00:53:43             |
|infersent(gru)              |   64       |   adam     |glove_cc_fix |   -         | 91.90 |82.4        |83.96   | 81.8        |83.30  |00:38:54             |
|indersent(bigru-last)       |   64       |   adam     |glove_cc_fix |   -         | 88.44 |81.3        |84.08   | 80.9        |83.64  |00:56:00             |
|infersent(bilstm-last)      |   64       |   adam     |glove_cc_fix |   -         | 89.75 |-           |84.27   | -           |83.63  |01:21:38             |
|infersent(inner-attention)  |   64       |   adam     |glove_cc_fix |   -         | 87.07 |82.3        |81.82   | 82.5        |82.23  |00:12:36             |
|infersent(hconv-net)        |   64       |   adam     |glove_cc_fix |   -         | 88.07 |83.7        |83.46   | 83.4        |83.23  |00:24:36             |
|esim                        |   32       |adam(0.0005)|glove_cc_tune|   92.6      | 90.81 | -          |87.55   | 88.0        |86.68  |11:03:31             |
|Siamese_BiLSTM              |   128      |   adam     |glove_cc_fix |   -         | -     | -          |83.47   | -           |83.22  |06:41:44             |
|Siamese_CNN                 |   128      |   adam     |glove_cc_tune|   -         | -     | -          |82.57   | -           |81.88  |00:33:51             |
|Siamese_IACNN               |            |            |             |   -         |       |            |        |             |       |                     |

- MedNLI

| model                      | batch_size | optimizer  | embedding                   | dev    | test  |train_time(1 TITAN X)|
|----------------------------|------------|------------|--------------------------   |--------|-------|---------------------|
|baseline                    |   -        |            |                             |76.0    |73.5   |                     |
|decomposable(intra-sentence)|   32       |   adam     |glove_cc_fix                 |73.40   |69.62  |00:01:02             |
|decomposable(intra-sentence)|   64       |   adam     |glove_cc_fix                 |69.96   |68.07  |00:00:34             |
|decomposable                |   32       |   adam     |glove_cc_fix                 |33.33   |33.33  |00:00:42             |
|infersent(lstm)             |   32       |   adam     |glove_cc_fix                 |75.84   |73.98  |00:41:32             |
|infersent(lstm)_feat        |   32       |   adam     |glove_cc_fix                 |76.20   |73.27  |01:04:31             |
|infersent(lstm)             |   32       |   adam     |elmo_id_elmo                 |75.77   |71.24  |01:43:23             |
|infersent(lstm)             |   32       |   adam     |glove_cc_fix elmo_id_elmo    |76.05   |73.70  |01:23:15             |
|infersent(lstm)             |   32       |   adam     |glove_cc_fix elmo_id_elmo_fix|76.20   |71.17  |01:12:26             |
|infersent(lstm)             |   64       |   adam     |glove_cc_fix                 |75.05   |72.78  |00:26:01             |
|infersent(lstm)_feat_scaled |   64       |   adam     |glove_cc_fix                 |76.92   |75.24  |00:20:55             |
|infersent(gru)              |   32       |   adam     |glove_cc_fix                 |75.48   |73.49  |00:31:01             |
|infersent(gru)_feat         |   32       |   adam     |glove_cc_fix                 |76.06   |73.98  |00:48:48             |
|infersent(gru)              |   32       |   adam     |elmo_id_elmo                 |76.63   |71.80  |00:56:56             |
|infersent(gru)              |   32       |   adam     |glove_cc_fix elmo_id_elmo    |77.71   |74.33  |01:41:11             |
|infersent(gru)              |   32       |   adam     |glove_cc_fix elmo_id_elmo_fix|75.77   |73.41  |01:18:55             |
|infersent(gru)              |   64       |   adam     |glove_cc_fix                 |76.42   |73.34  |00:17:54             |
|infersent(gru)_feat_scaled  |   64       |   adam     |glove_cc_fix                 |76.42   |74.61  |00:18:33             |
|infersent(bilstm-last)      |   32       |   adam     |glove_cc_fix                 |76.70   |72.86  |03:00:21             |
|infersent(bilstm-last)_feat |   32       |   adam     |glove_cc_fix                 |75.84   |74.40  |02:02:38             |
|infersent(bilstm-last)      |   32       |   adam     |elmo_id_elmo                 |75.34   |73.28  |02:06:59             |
|infersent(bilstm-last)      |   32       |   adam     |glove_cc_fix elmo_id_elmo    |76.13   |72.86  |02:32:00             |
|infersent(bilstm-last)      |   32       |   adam     |glove_cc_fix elmo_id_elmo_fix|75.34   |72.71  |02:42:15             |
|infersent(bilstm-last)      |   64       |   adam     |glove_cc_fix                 |75.41   |73.63  |01:28:09             |
|infersent(bilstm-last)_feat_scaled|   64 |   adam     |glove_cc_fix                 |76.34   |73.63  |00:36:32             |
|infersent(bigru-last)       |   32       |   adam     |glove_cc_fix                 |76.05   |74.40  |01:22:44             |
|infersent(bigru-last)_feat  |   32       |   adam     |glove_cc_fix                 |76.85   |73.28  |01:02:17             |
|infersent(bigru-last)       |   32       |   adam     |elmo_id_elmo                 |77.42   |73.07  |02:35:59             |
|infersent(bigru-last)       |   32       |   adam     |glove_cc_fix elmo_id_elmo    |77.13   |74.26  |02:02:54             |
|infersent(bigru-last)       |   32       |   adam     |glove_cc_fix elmo_id_elmo_fix|77.42   |74.54  |02:09:25             |
|infersent(bigru-last)       |   64       |   adam     |glove_cc_fix                 |75.13   |72.43  |00:39:59             |
|infersent(bigru-last)_feat_scaled|   64  |   adam     |glove_cc_fix                 |76.34   |72.43  |00:33:42             |
|infersent(bilstm-max)       |   32       |   adam     |glove_cc_fix                 |77.20   |74.68  |01:29:12             |
|infersent(bilstm-max)_feat  |   32       |   adam     |glove_cc_fix                 |77.49   |75.18  |02:49:01             |
|infersent(bilstm-max)       |   32       |   adam     |elmo_id_elmo                 |76.70   |73.07  |02:30:09             |
|infersent(bilstm-max)       |   32       |   adam     |glove_cc_fix elmo_id_elmo    |78.06   |74.47  |02:24:56             |
|infersent(bilstm-max)       |   32       |   adam     |glove_cc_fix elmo_id_elmo_fix|77.85   |73.28  |03:38:43             |
|infersent(bilstm-max)       |   64       |   adam     |glove_cc_fix                 |77.20   |75.74  |00:57:01             |
|infersent(bilstm-max)_feat  |   64       |   adam     |glove_cc_fix                 |77.99   |75.60  |01:08:24             |
|infersent(bilstm-max)_feat_scaled|   64  |   adam     |glove_cc_fix                 |77.85   |75.87  |00:44:21             |
|infersent(bilstm-max)       |   64       | adam(clr)  |glove_cc_fix                 |78.06   |76.93  |01:00:24             |
|infersent(bilstm-mean)      |   32       |   adam     |glove_cc_fix                 |77.78   |74.12  |01:10:32             |
|infersent(bilstm-mean)      |   32       |   adam     |glove_cc_fix                 |76.20   |74.26  |01:16:55             |
|infersent(bilstm-mean)      |   32       |   adam     |elmo_id_emlo                 |76.27   |73.70  |02:17:17             |
|infersent(bilstm-mean)      |   32       |   adam     |glove_cc_fix elmo_id_elmo    |77.63   |74.40  |02:24:55             |
|infersent(bilstm-mean)      |   32       |   adam     |glove_cc_fix elmo_id_elmo_fix|77.92   |72.78  |05:06:39             |
|infersent(bilstm-mean)      |   64       |   adam     |glove_cc_fix                 |76.77   |**76.37**|01:07:43           |
|infersent(bilstm-mean)_feat_scaled|   64 |   adam     |glove_cc_fix                 |77.28   |72.29  |01:07:43             |
|infersent(inner-attention)  |   32       |   adam     |glove_cc_fix                 |71.89   |70.11  |00:01:08             |
|infersent(inner_attention)  |   32       |   adam     |glove_cc_fix                 |72.97   |72.15  |00:00:43             |
|infersent(inner-attention)  |   64       |   adam     |glove_cc_fix                 |73.05   |71.66  |00:00:44             |
|infersent(hcnn)             |   32       |   adam     |glove_cc_fix                 |74.98   |75.11  |00:02:18             |
|infersent(hcnn)_feat        |   32       |   adam     |glove_cc_fix                 |75.77   |73.35  |00:02:15             |
|infersent(hcnn)             |   32       |   adam     |elmo_id_elmo                 |74.55   |72.15  |00:39:35             |
|infersent(hcnn)             |   32       |   adam     |glove_cc_fix elmo_id_elmo    |75.05   |73.91  |00:34:36             |
|infersent(hcnn)             |   32       |   adam     |glove_cc_fix elmo_id_elmo_fix|75.99   |72.15  |00:45:27             |
|infersent(hcnn)             |   64       |   adam     |glove_cc_fix                 |74.55   |73.70  |00:02:09             |
|infersent(hcnn)             |   128      |   adam     |glove_cc_fix                 |75.91   |75.17  |00:01:57             |
|infersent(hcnn)_feat_scaled |   64       |   adam     |glove_cc_fix                 |76.48   |74.54  |00:02:45             |
|esim                        |   32       |   adam     |glove_cc_fix                 |77.49   |74.75  |03:20:20             |
|esim_feat                   |   32       |   adam     |glove_cc_fix                 |77.78   |75.88  |04:10:12             |
|esim                        **|   32       |   adam     |glove_cc_fix elmo_id_elmo    |77.92 |**76.23**|03:15:43             |
|esim                        |   32       |   adam     |glove_cc_fix elmo_id_elmo_fix|77.49   |75.04  |03:08:38             |
|esim                        |   64       |   adam     |glove_cc_fix                 |78.42   |73.98  |01:41:50             |
|esim                        |   64       |   adam     |glove_cc_fix                 |77.49   |75.53  |01:46:02             |
|esim_feat_scaled            |   64       |   adam     |glove_cc_fix                 |77.13   |73.56  |01:23:32             |
|siamese_BiLSTM              |   32       |   adam     |glove_cc_fix                 |75.12   |73.49  |03:06:36             |
|siamese_BiLSTM_feat         |   32       |   adam     |glove_cc_fix                 |75.20   |73.55  |01:39:17             |
|siamese_BiLSTM              |   32       |   adam     |glove_cc_fix elmo_id_elmo    |75.19   |73.34  |06:04:32             |
|siamese_BiLSTM              |   32       |   adam     |glove_cc_fix elmo_id_elmo_fix|75.84   |73.27  |02:29:43             |
|siamese_BiLSTM              |   64       |   adam     |glove_cc_fix                 |74.62   |71.52  |01:07:20             |
|siamese_BiLSTM_feat_scaled  |   64       |   adam     |glove_cc_fix                 |75.34   |73.14  |01:07:20             |
|siamese_CNN                 |   32       |   adam     |glove_cc_fix                 |72.83   |70.39  |00:02:56             |
|siamese_CNN                 |   32       |   adam     |glove_cc_fix elmo_id_elmo    |73.83   |72.01  |01:02:55             |
|siamese_CNN                 |   32       |   adam     |glove_cc_fix elmo_id_elmo_fix|73.90   |71.16  |01:12:29             |
|siamese_CNN                 |   64       |   adam     |glove_cc_fix                 |73.90   |70.67  |00:02:39             |
|siamese_CNN_feat_scaled     |   64       |   adam     |glove_cc_fix                 |74.69   |72.22  |00:02:39             |
|siamese_IACNN               |   32       |   adam     |glove_cc_fix                 |33.33   |33.33  |00:02:24             |
|bert_fine_tuning            |   24       |   -        |-                            |48.88   |-      |-                    |

- Conclusion of MedNLI Experiments

1. use a small batch size
2. fasttext_cc performs worse than glove_cc
3. fixing glove_cc is slightly better than fine tuning glove_cc
4. Siamese_IACNN and Decomposable with intra-sentence suck, always

### Reference

- Part of my code are based on [mednil](https://github.com/jgc128/mednli). Great work, thanks!
