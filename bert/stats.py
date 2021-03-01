from nltk import sent_tokenize, word_tokenize
from collections import defaultdict
import json
import pandas as pd
from nltk.stem.porter import *
import sentencepiece as spm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib import rc, font_manager
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from collections import defaultdict
from scipy.stats.stats import spearmanr
import random




def file_to_df(input_path, classification):
    all_docs = []
    counter = 0
    num_words = 0
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f:
            counter += 1
            if counter % 10000 == 0:
                print('Processing json: ', counter)
            line = json.loads(line)
            title = line.get('title') or ''
            abstract = line.get('abstract') or ''

            text = title + '. ' + abstract

            if not classification:
                fulltext = line.get("fulltext") or ''
                text = text + ' ' + fulltext

            num_words += len(text.split())
            try:
                kw = line['keywords']
            except:
                kw = line['keyword']
            if isinstance(kw, list):
                kw = ";".join(kw)

            all_docs.append([text,kw])

    df = pd.DataFrame(all_docs)
    df.columns = ["text", "keyword"]
    print(input_path, 'data size: ', df.shape)
    print('Avg words: ', num_words/df.shape[0])
    return df




class Stats(object):
    def __init__(self, df_test, bpe, bpe_model_path):
        self.bpe = bpe
        if self.bpe:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(bpe_model_path)

        self.get_counts(df_test, max_length=256)

    def preprocess_line(self, line):
        words = []


        text = line['text']
        text = text.replace('-', ' ')
        text = text.replace('/', ' ')
        text = text.replace('∗', ' ')
        for sent in sent_tokenize(text):
            sent = word_tokenize(sent)
            if self.bpe:
                bpe_sent = []
                for w in sent:
                    w = w.lower()
                    bpe_word = self.sp.EncodeAsPieces(w)
                    bpe_sent.append(bpe_word)
                    words.extend(bpe_word)
                words.append('<eos>')
            else:
                words.extend([w.lower() for w in sent] + ['<eos>'])
        return words

    def get_counts(self, df, max_length):
        stemmer = PorterStemmer()
        stemmed_string = ""

        docs = []
        for idx, line in df.iterrows():
            words = self.preprocess_line(line)
            stems = " ".join([stemmer.stem(w.lower()) for w in words])
            stemmed_string += stems + " "

            tokenized_keywords = []
            keywords = line['keyword'].lower()
            keywords = keywords.replace('-', ' ')
            keywords = keywords.replace('/', ' ')
            keywords = keywords.replace('∗', ' ')

            for kw in keywords.split(';'):
                if not self.bpe:
                    kw = kw.split()
                else:
                    kw = self.sp.EncodeAsPieces(kw)
                tokenized_keywords.append(kw)
            docs.append([words, tokenized_keywords])

        counts = defaultdict(int)

        for i, doc in enumerate(docs):
            words, kws = doc
            length = len(words)

            for j, word in enumerate(words):
                for kw in kws:
                    lkw = len(kw)

                    is_keyword = False
                    if j + lkw < length:
                        for k in range(lkw):
                            w = words[j + k]
                            if stemmer.stem(w.lower()) != stemmer.stem(kw[k].lower()):
                                break


                        else:
                            is_keyword = True
                    if is_keyword:
                        for k in range(lkw):
                            if j + k < max_length:
                                counts[j] += 1

        counts = sorted(counts.items(), key=lambda x:x[0])
        print("Counts: ", counts)


        import numpy as np

        font_size = 12
        stemmer = PorterStemmer()

        # 'family': 'serif', 'serif': ['Computer Modern Roman'],

        font_properties = {'weight': 'normal', 'size': font_size}

        font_manager.FontProperties(family='Computer Modern Roman', style='normal',
                                    size=font_size, weight='normal', stretch='normal')
        rc('text', usetex=True)
        rc('font', **font_properties)


        sat = np.array([x[1] for x in counts])
        xdom = np.arange(0, len(sat), 1)

        ax1 = sns.barplot(xdom, sat, color="black")

        ax1.set_yticks([0,10,20,30,40,50,60,70])
        #print()
        ax1.set_xticks(list(range(0,256,32)))
        ax1.set(xlabel='Position', ylabel='Number of keywords')

        fig = ax1.get_figure()
        fig.savefig("semeval_kw_distrib.png")

def get_tfidf():
    all_att_list = []
    all_tfidf_list = []
    best_att = []
    best_tfidf = []
    with open('attentionviz/att_object.pickle', 'rb') as file:
        att_file = pickle.load(file)
        documents = []
        for example in tqdm.tqdm(att_file):
            text = example['Text']
            cleaned_text = []
            for idx, word in enumerate(text):
                if '▁' in word and word not in ['<eos>', '<pad>']:
                    word = word.replace('▁', '')
                    if word == 'd':
                        word = word + '-'
                if len(word) > 0:
                    cleaned_text.append(word)
            text = cleaned_text
            documents.append(" ".join(text))

            matrix = example["AttentionMatrix"]
            atMat = np.sum(matrix, axis=0)
            diag = atMat.diagonal()
            att_dict = defaultdict(list)
            for idx, w in enumerate(text):
                att_dict[w].append(diag[idx])
            att_list = []
            for w, att_l in att_dict.items():
                att = sum(att_l)/len(att_l)
                att_list.append((w, att))
            att_list = sorted(att_list, key=lambda x: x[0], reverse=False)
            all_att_list.append([x[1] for x in att_list])
            att_list = sorted(att_list, key=lambda x: x[1], reverse=False)
            random.shuffle(att_list)
            best_att.append([x[0] for x in att_list[:10]])
            '''for w, s in att_list:
                print(w, s)
            print('-----------------------------------------------')
            print()'''
        #print()

    vect = TfidfVectorizer(tokenizer=lambda x: x.split())

    tfidf_matrix = vect.fit_transform(documents)
    feature_names = vect.get_feature_names()
    for doc_num in range(len(documents)):
        feature_index = tfidf_matrix[doc_num, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[doc_num, x] for x in feature_index])
        tfidf_scores = sorted(tfidf_scores, key=lambda x: x[0], reverse=False)

        tfidf_list = []
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            #print(w, s)
            tfidf_list.append((w,s))

        all_tfidf_list.append([x[1] for x in tfidf_list])

        tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=False)
        random.shuffle(tfidf_scores)

        tfidf_list = []
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            # print(w, s)
            tfidf_list.append((w, s))

        best_tfidf.append([x[0] for x in tfidf_list[:10]])

    for att, tfidf in zip(all_att_list, all_tfidf_list):
        print(att)
        print(tfidf)
        print(len(att), len(tfidf))
        spearman = spearmanr(att, tfidf)
        print("Spearman: ", spearman)

    intersections = []
    for att, tfidf in zip(best_att, best_tfidf):
        inter = list(set(att) & set(tfidf))
        intersections.append(len(inter))
        print(att)
        print(tfidf)
        print(len(inter))
    print(sum(intersections)/len(intersections))







if __name__ == '__main__':
    get_tfidf()

