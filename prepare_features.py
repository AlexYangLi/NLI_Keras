# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: prepare_features.py

@time: 2019/3/29 21:47

@desc:

"""

import os
import networkx as nx
from gensim import corpora
from gensim.models import TfidfModel
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from config import TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, TEST_DATA_TEMPLATE, PROCESSED_DATA_DIR, FEATURE_DIR, \
    TRAIN_FEATURES_TEMPLATE, DEV_FEATURES_TEMPLATE, TEST_FEATURES_TEMPLATE
from utils.io import format_filename, pickle_load, pickle_dump
from utils.features import *


class Feature(object):
    def __init__(self, genre):
        self.genre = genre
        self.train_data = pickle_load(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, genre))
        self.dev_data = pickle_load(format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, genre))
        self.test_data = pickle_load(format_filename(PROCESSED_DATA_DIR, TEST_DATA_TEMPLATE, genre))

        if not os.path.exists(FEATURE_DIR):
            os.makedirs(FEATURE_DIR)

    def gen_all_features(self, data_type, scaled=False):
        if scaled:
            feat_file = self.format_feature_file(data_type, 'all_scaled')
        else:
            feat_file = self.format_feature_file(data_type, 'all')
        if os.path.exists(feat_file):
            features = pickle_load(feat_file)
        else:
            features = list()
            feat_types = [('len_dis', length_distance), ('lcs_seq', lcs_seq_norm), ('lcs_str', lcs_str_1_norm),
                          ('edit_dist', edit_distance), ('jaro', jaro_distance), ('jaro_winkler', jaro_winkler_dist),
                          ('fuzz', fuzzy), ('simhash', simhash), ('w_share', word_share),
                          ('w_ngram_dist', word_ngram_distance), ('c_ngram_ol', char_ngram_overlap),
                          ('w_ngram_ol', word_ngram_overlap)]
            for feat_type, sim_func in feat_types:
                features.append(self.add_similarity_feature(data_type, feat_type, sim_func))

            features.append(self.add_weighted_word_ngram_overlap_feature(data_type))
            features.append(self.add_tfidf_feature(data_type))
            features.append(self.add_word_power_feature(data_type))
            features.append(self.add_graph_feature(data_type))
            features = np.concatenate(features, axis=-1)

            if scaled:
                scaler = StandardScaler()
                features = scaler.fit_transform(features)
                joblib.dump(scaler, os.path.join(FEATURE_DIR, '{}_scaler.model'.format(self.genre)))

            pickle_dump(feat_file, features)

        print('Logging Info - {} : all feature shape : {}'.format(data_type, features.shape))

    def add_similarity_feature(self, data_type, feat_type, sim_func):
        feat_file = self.format_feature_file(data_type, feat_type)
        if os.path.exists(feat_file):
            features = pickle_load(feat_file)
        else:
            len_dist_feat = np.array([sim_func(p, h) for p, h in zip(self.get_data(data_type)['premise'],
                                                                     self.get_data(data_type)['hypothesis'])])
            features = self.check_and_expand_shape(len_dist_feat)
            pickle_dump(feat_file, features)
        print('Logging Info - {} : {} feature shape : {}'.format(data_type, feat_type, features.shape))
        return features

    def add_tfidf_feature(self, data_type):
        feat_file = self.format_feature_file(data_type, 'tfidf')
        if os.path.exists(feat_file):
            features = pickle_load(feat_file)
        else:
            dictionary, tfidf_model = self.tfidf_model()
            features = list()
            for premise, hypothesis in zip(self.get_data(data_type)['premise'], self.get_data(data_type)['hypothesis']):
                premise = premise.split()
                hypothesis = hypothesis.split()
                p_tfidf = dict(tfidf_model[dictionary.doc2bow(premise)])
                h_tfidf = dict(tfidf_model[dictionary.doc2bow(hypothesis)])
                features.append([np.sum(list(p_tfidf.values())), np.sum(list(h_tfidf.values())),
                                 np.mean(list(p_tfidf.values())), np.mean(list(h_tfidf.values()))])
            features = np.array(features)
            pickle_dump(feat_file, features)
        print('Logging Info - {} : w_ngram_ol_tfidf feature shape : {}'.format(data_type, features.shape))
        return features

    def add_weighted_word_ngram_overlap_feature(self, data_type):
        feat_file = self.format_feature_file(data_type, 'w_ngram_ol_tfidf')
        if os.path.exists(feat_file):
            features = pickle_load(feat_file)
        else:
            dictionary, tfidf_model = self.tfidf_model()
            idf_model = tfidf_model.idfs
            features = list()
            for premise, hypothesis in zip(self.get_data(data_type)['premise'], self.get_data(data_type)['hypothesis']):
                premise = premise.split()
                p_tfidf = dict(tfidf_model[dictionary.doc2bow(premise)])
                input_premise = [(word, idf_model.get(dictionary.token2id.get(word, 0), 0.0),
                                  p_tfidf.get(dictionary.token2id.get(word, 0), 0.0)) for word in premise]

                hypothesis = hypothesis.split()
                h_tfidf = dict(tfidf_model[dictionary.doc2bow(hypothesis)])
                input_hypothesis = [(word, idf_model.get(dictionary.token2id.get(word, 0), 0.0),
                                     h_tfidf.get(dictionary.token2id.get(word, 0), 0.0)) for word in hypothesis]
                features.append(weighted_word_ngram_overlap(input_premise, input_hypothesis))
            features = np.array(features)
            pickle_dump(feat_file, features)
        print('Logging Info - {} : w_ngram_ol_tfidf feature shape : {}'.format(data_type, features.shape))
        return features

    def add_word_power_feature(self, data_type):
        feat_file = self.format_feature_file(data_type, 'word_power')
        if os.path.exists(feat_file):
            features = pickle_load(feat_file)
        else:
            power_word = self.get_power_word()
            num_least = 100
            features = list()
            for premise, hypothesis in zip(self.get_data(data_type)['premise'], self.get_data(data_type)['hypothesis']):
                premise = premise.split()
                hypothesis = hypothesis.split()

                rate = [1.0, 1.0]
                share_words = list(set(premise).intersection(set(hypothesis)))
                for word in share_words:
                    if word not in power_word:
                        continue
                    if power_word[word][0] * power_word[word][5] < num_least:   # 共享词出现在双侧语句对数量要大于num_least
                        continue
                    rate[0] *= (1.0 - power_word[word][6])  # 共享词但是语句对不是正确的（label!=2）
                p_diff = list(set(premise).difference(set(hypothesis)))
                h_diff = list(set(premise).difference(set(hypothesis)))
                all_diff = set(p_diff + h_diff)
                for word in all_diff:
                    if word not in power_word:
                        continue
                    if power_word[word][0] * power_word[word][3] < num_least:   # 共享词只出现在单侧语句数量要大于num_least
                        continue
                    rate[1] *= (1.0 - power_word[word][4])  # 非共享词但是语句对是正确的（label=2）
                rate = [1 - num for num in rate]
                features.append(rate)
            features = np.array(features)
            pickle_dump(feat_file, features)
        print('Logging Info - {} : word_power feature shape : {}'.format(data_type, features.shape))
        return features

    def add_graph_feature(self, data_type):
        feat_file = self.format_feature_file(data_type, 'word_power')
        if os.path.exists(feat_file):
            graph_features = pickle_load(feat_file)
        else:
            sent2id, graph = self.generate_graph()

            n2clique = {}
            cliques = []
            for clique in nx.find_cliques(graph):
                for n in clique:
                    if n not in n2clique:
                        n2clique[n] = []
                    n2clique[n].append(len(cliques))
                cliques.append(clique)

            n2cc = {}
            ccs = []
            for cc in nx.connected_components(graph):
                for n in cc:
                    n2cc[n] = len(ccs)
                ccs.append(cc)

            pagerank = nx.pagerank(graph, alpha=0.9, max_iter=100)

            hits_h, hits_a = nx.hits(graph, max_iter=100)

            indegree_features = list()
            clique_features = list()
            cc_features = list()
            pagerank_features = list()
            hits_features = list()
            shortestpath_features = list()
            # neighbor_features = list()
            for premise, hypothesis in zip(self.get_data(data_type)['premise'], self.get_data(data_type)['hypothesis']):
                p_id = sent2id[premise]
                h_id = sent2id[hypothesis]

                # graph in-degree fetures
                indegree_features.append([graph.degree[p_id], graph.degree[h_id]])

                # clique features
                edge_max_clique_size = 0
                num_clique = 0
                for clique_id in n2clique[p_id]:
                    if h_id in cliques[clique_id]:
                        edge_max_clique_size = max(edge_max_clique_size, len(cliques[clique_id]))
                        num_clique += 1
                clique_features.append([edge_max_clique_size, num_clique])

                lnode_max_clique_size = 0
                rnode_max_clique_size = 0
                for clique_id in n2clique[p_id]:
                    lnode_max_clique_size = max(lnode_max_clique_size, len(cliques[clique_id]))

                for clique_id in n2clique[h_id]:
                    rnode_max_clique_size = max(rnode_max_clique_size, len(cliques[clique_id]))

                clique_features[-1] += [lnode_max_clique_size, rnode_max_clique_size,
                                        max(lnode_max_clique_size, rnode_max_clique_size),
                                        min(lnode_max_clique_size, rnode_max_clique_size)]

                # connected components features
                cc_features.append([len(ccs[n2cc[p_id]])])

                # page rank features
                pr1 = pagerank[p_id] * 1e6
                pr2 = pagerank[h_id] * 1e6
                pagerank_features.append([pr1, pr2, max(pr1, pr2), min(pr1, pr2), (pr1 + pr2) / 2.])

                # graph hits features
                h1 = hits_h[p_id] * 1e6
                h2 = hits_h[h_id] * 1e6
                a1 = hits_a[p_id] * 1e6
                a2 = hits_a[h_id] * 1e6
                hits_features.append([h1, h2, a1, a2, max(h1, h2), max(a1, a2), min(h1, h2), min(a1, a2),
                                      (h1 + h2) / 2., (a1 + a2) / 2.])

                # graph shortest path features
                shortest_path = -1
                weight = graph[p_id][h_id]['weight']
                graph.remove_edge(p_id, h_id)
                if nx.has_path(graph, p_id, h_id):
                    shortest_path = nx.dijkstra_path_length(graph, p_id, h_id)
                graph.add_edge(p_id, h_id, weight=weight)
                shortestpath_features.append([shortest_path])

                # graph neighbour features
                # l = []
                # r = []
                # l_nb = graph.neighbors(p_id)
                # r_nb = graph.neighbors(h_id)
                # for n in l_nb:
                #     if (n != h_id) and (n != p_id):
                #         l.append(graph[p_id][n]['weight'])
                # for n in r_nb:
                #     if (n != h_id) and (n != p_id):
                #         r.append(graph[h_id][n]['weight'])
                # if len(l) == 0 or len(r) == 0:
                #     neighbor_features.append([0.0] * 11)
                # else:
                #     neighbor_features.append(l + r +
                #                              [len(list((set(l_nb).union(set(r_nb))) ^ (set(l_nb) ^ set(r_nb))))])

            graph_features = np.concatenate((np.array(indegree_features), np.array(clique_features),
                                             np.array(cc_features), np.array(pagerank_features),
                                             np.array(hits_features), np.array(shortestpath_features)), axis=-1)
            pickle_dump(feat_file, graph_features)
        print('Logging Info - {} : graph feature shape : {}'.format(data_type, graph_features.shape))
        return graph_features

    def format_feature_file(self, data_type, feat_type):
        if data_type == 'train':
            feat_file = format_filename(FEATURE_DIR, TRAIN_FEATURES_TEMPLATE, self.genre, feat_type)
        elif data_type == 'dev' or data_type == 'valid':
            feat_file = format_filename(FEATURE_DIR, DEV_FEATURES_TEMPLATE, self.genre, feat_type)
        elif data_type == 'test':
            feat_file = format_filename(FEATURE_DIR, TEST_FEATURES_TEMPLATE, self.genre, feat_type)
        else:
            raise ValueError('Data Type `{}` not understood'.format(data_type))
        return feat_file

    def get_data(self, data_type):
        if data_type == 'train':
            return self.train_data
        elif data_type == 'valid' or data_type == 'dev':
            return self.dev_data
        elif data_type == 'test':
            return self.test_data
        else:
            raise ValueError('Data Type `{}` not understood'.format(data_type))

    @staticmethod
    def check_and_expand_shape(inputs):
        if len(inputs.shape) == 1:
            return np.expand_dims(inputs, -1)
        else:
            return inputs

    def tfidf_model(self):
        print('Logging Info - Get Tf-idf model...')
        tfidf_model_path = os.path.join(FEATURE_DIR, '{}_tfidf.model').format(self.genre)
        dict_path = os.path.join(FEATURE_DIR, '{}_tfidf.dict').format(self.genre)
        if os.path.exists(tfidf_model_path):
            dictionary = pickle_load(dict_path)
            tfidf_model = TfidfModel.load(tfidf_model_path)
        else:
            corpus = [text.split() for text in
                      self.train_data['premise'] + self.train_data['hypothesis'] + self.dev_data['premise'] +
                      self.dev_data['hypothesis'] + self.test_data['premise'] + self.test_data['hypothesis']]
            dictionary = corpora.Dictionary(corpus)
            corpus = [dictionary.doc2bow(text) for text in corpus]
            tfidf_model = TfidfModel(corpus)

            del corpus
            tfidf_model.save(tfidf_model_path)
            pickle_dump(dict_path, dictionary)

        return dictionary, tfidf_model

    def get_power_word(self):
        """
        计算数据中词语的影响力，格式如下：
        词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，
                 5. 双侧语句对比例，6. 双侧语句对正确比例]
        """
        print('Logging Info - Get power word...')
        words_power_path = os.path.join(FEATURE_DIR, '{}_power_word.pkl'.format(self.genre))
        if os.path.exists(words_power_path):
            words_power = pickle_load(words_power_path)
        else:
            words_power = {}
            x_a = [text.split() for text in
                   self.train_data['premise'] + self.dev_data['premise'] + self.test_data['premise']]
            x_b = [text.split() for text in
                   self.train_data['hypothesis'] + self.dev_data['hypothesis'] + self.test_data['hypothesis']]
            y = self.train_data['label'] + self.dev_data['label'] + self.test_data['label']
            for i in range(len(x_a)):
                label = y[i]
                q1_words = x_a[i]
                q2_words = x_b[i]
                all_words = set(q1_words + q2_words)
                q1_words = set(q1_words)
                q2_words = set(q2_words)
                for word in all_words:
                    if word not in words_power:
                        words_power[word] = [0. for _ in range(7)]
                    words_power[word][0] += 1.  # 计算出现语句对的数量
                    words_power[word][1] += 1.  # 计算出现语句对比例

                    if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                        words_power[word][3] += 1.      # 计算单侧语句对比例
                        if 0 == label:
                            words_power[word][2] += 1.  # 计算正确语句对比例
                            words_power[word][4] += 1.  # 计算单侧语句正确比例
                    if (word in q1_words) and (word in q2_words):
                        words_power[word][5] += 1.  # 计算双侧语句数量
                        if 2 == label:
                            words_power[word][2] += 1.  # 计算正确语句对比例
                            words_power[word][6] += 1.  # 计算双侧语句正确比例

            for word in words_power:
                words_power[word][1] /= len(x_a)    # 计算出现语句对比例=出现语句对数量/总的语句对数量
                words_power[word][2] /= words_power[word][0]    # 计算正确语句对比例=正确语句对数量/出现语句对数量
                if words_power[word][3] > 1e-6:
                    words_power[word][4] /= words_power[word][3]    # 计算单侧语句正确比例=单侧语句正确数量/出现单侧语句数量
                words_power[word][3] /= words_power[word][0]    # 计算出现单侧语句对比例=出现单侧语句数量/出现语句对数量
                if words_power[word][5] > 1e-6:
                    words_power[word][6] /= words_power[word][5]    # 计算双侧语句正确比例=双侧语句正确数量/出现双侧语句数量
                words_power[word][5] /= words_power[word][0]    # 计算出现双侧语句对比例=出现双侧语句数量/出现语句数量
            del x_a, x_b, y
            pickle_dump(words_power_path, words_power)

        return words_power

    def generate_graph(self):
        print('Logging Info - Get graph...')
        sent2id_path = os.path.join(FEATURE_DIR, '{}_graph_sent2id.pkl'.format(self.genre))
        graph_path = os.path.join(FEATURE_DIR, '{}_graph.pkl'.format(self.genre))
        if os.path.exists(graph_path):
            sent2id = pickle_load(sent2id_path)
            graph = pickle_load(graph_path)
        else:
            sent2id = {}    # sentence to id
            graph = nx.Graph()
            for data_type in ['train', 'dev', 'test']:
                for premise, hypothesis in zip(self.get_data(data_type)['premise'], self.get_data(data_type)['hypothesis']):
                    if premise not in sent2id:
                        sent2id[premise] = len(sent2id)
                    if hypothesis not in sent2id:
                        sent2id[hypothesis] = len(sent2id)
                    p_id = sent2id[premise]
                    h_id = sent2id[hypothesis]

                    match = 0.0
                    premise = premise.split()
                    hypothesis = hypothesis.split()
                    for w1 in premise:
                        if w1 in hypothesis:
                            match += 1

                    if len(premise) + len(hypothesis) == 0:
                        weight = 0.0
                    else:
                        weight = 2.0 * (match / (len(premise) + len(hypothesis)))
                    graph.add_edge(p_id, h_id, weight=weight)
            pickle_dump(sent2id_path, sent2id)
            pickle_dump(graph_path, graph)
        return sent2id, graph


if __name__ == '__main__':
    feature = Feature('mednli')
    # feature.gen_all_features('train')
    # feature.gen_all_features('dev')
    # feature.gen_all_features('test')
    feature.gen_all_features('train', True)
    feature.gen_all_features('dev', True)
    feature.gen_all_features('test', True)





