#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:30:04 2020

https://open.spotify.com/album/2XvByxHjZuNhT2SRT8ofMV

@author: bosko
"""

import config
import os
import glob
import json
from tqdm import tqdm
from eval import eval
from lemmagen3 import Lemmatizer
from sklearn import cluster
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn import cluster
from gensim.test.utils import datapath
import string
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer 
import random
import re
import pickle
from stopwordsiso import stopwords

from tf_idf_solve import predict, prepare, extract_naive, load, save
def build_kw(path, lang='et'):
    outs = {}
    sw = stopwords(lang)
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip("\n")
            parsed = prepare(line, lang)        
            if not parsed in sw:
                outs[parsed] = outs.get(parsed,[]) + [line]
    return outs


def build_kw_json(path, lang='et'):
    outs = {}
    sw = stopwords(lang)
    df = pd.read_csv(path, names=["kw"], dtype={})
    kw_df = df["kw"].astype(str).tolist()
    del df
    kws_ = set()
    for kw in tqdm(kw_df):
        #if lang == 'hr':
        #    kws_.add(kw)
        #else:
        for k in kw.split(';'):
            kws_.add(k.lower())
    return set(kws_)
    for line in list(kws_):
        parsed = prepare(line, lang)        
        if not parsed in sw:
            outs[parsed] = outs.get(parsed,[]) + [line]
    return outs
tqdm.pandas()
def parse_tags(file="predictions/russian_predictions.csv"):
    data = pd.read_csv(file, encoding='utf-8') 
    data = data.fillna(" ")
    abstracts = data["abstract"] + " " + data["title"]
    out_data = {
        "text+title" : abstracts,
        "original_tags": data["keywords_in_text"].to_list(), 
        "tnt_tags": data["predicted"].to_list()}        
    return out_data

def parse_jsons(file):
    #{'title':title, 'abstract':abstract, 'keywords':keywords,'keywords_org' : keywords_org}
    file_dicts = []
    with open(file, "r", encoding="utf-8") as f:
        f = f.readlines()
        for line in f:
            line = line.strip()
            dic = dict(json.loads(line))
            file_dicts.append(dic)
    df_files = pd.DataFrame(file_dicts)
    df_files["text+title"] = df_files["abstract"] + " " + df_files["title"]
    del df_files["abstract"]
    del df_files["title"]
    print(df_files['keywords'])
    return df_files

def learn_json(file, path_tags, lang, opt = ""):
    data = parse_jsons(file)
    lang_prepare = lang if opt == 'nat' else 'ru'
    outs = build_kw(path_tags, lang_prepare)
    print(outs)
    #PREPARE
    tfidf_vectorizer=TfidfVectorizer(use_idf=True, vocabulary = list(outs.keys()), ngram_range = (1,2))    
    processed = data["text+title"].progress_apply(prepare, lang = lang_prepare) #map(lambda txt: prepare(txt))

    _ = tfidf_vectorizer.fit_transform(processed)   

    save(outs,tfidf_vectorizer,lang+"_"+opt)    

def learn(prediction_path = "predictions/croatian_predictions.csv", tags_path = "tags/Ekspress_tagid-cyrl..csv", lang="ru"):
    #Data
    data = parse_tags(prediction_path)
    #Taglists
    taglist = build_kw("tags/Ekspress_tagid-latin.csv" if lang == 'et' else "tags/Ekspress_tagid-cyrl.csv", lang= lang)
    outs = taglist
    tfidf_vectorizer=TfidfVectorizer(use_idf=True, vocabulary = list(outs.keys()), ngram_range = (1,2))
    
    processed = data["text+title"].apply(prepare) #map(lambda txt: prepare(txt))

    _ = tfidf_vectorizer.fit_transform(processed)   
    #export
    print("EXPORTING")
    save(outs,tfidf_vectorizer, lang)    
    print("EXPORTING DONE")
    
   
def test(lang='lv',opt='nat'):
    df = parse_jsons("data/hr/nat_test.json")
    df["tfidf_tags"] =  df["text+title"].progress_apply(predict, lang=lang+"_"+opt)    
    #df["tfidf_tags"] =  df["tfidf_tags"].progress_apply(";".join)

    path_new_predictions = os.path.join("predictions",lang+"_"+opt+".csv")
    df.to_csv(path_new_predictions, index=False)    

def test_csv(path="predictions/croatian_predictions.csv", lang="hr", opt = "nat"):
    df = pd.read_csv(path, encoding='utf-8')
    print(df.columns)
    df["text+title"] = df["title"] + df["abstract"]
    df["tfidf_tags"] =  df["text+title"].progress_apply(predict, lang=lang+"_"+opt)    
    del df["text+title"] 
    print(df["tfidf_tags"])
    df.to_csv("final_"+lang+"_"+opt+".csv", encoding='utf-8')

for lang in ['hr']:
    for opt in ["nat"]:
        learn_json(file="data/"+lang+"/"+opt+"_valid.json", path_tags = "data/"+lang+"/"+opt+"_train_keywords.csv", lang=lang, opt=opt)#"nat")
        test(lang=lang, opt=opt)