
from nltk.stem.porter import *
from nltk import word_tokenize, sent_tokenize
import os
import yake
import argparse
from preprocessing import file_to_df
from eval_seq2seq import eval
from lemmagen3 import Lemmatizer
import pke


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='data/croatian/croatian_test.json',
                        help='paths to datasets separated with ;')
    parser.add_argument('--lang', type=str, default='croatian', help='language')
    parser.add_argument('--num_keywords', type=int, default=10, help='Number of keywords')
    args = parser.parse_args()

    if args.lang == 'english':
        stemmer = PorterStemmer()
    elif args.lang == 'estonian':
        stemmer = Lemmatizer('et')
    elif args.lang == 'croatian':
        stemmer = Lemmatizer('hr')

    language = args.lang
    numOfKeywords = args.num_keywords
    input_paths = args.datasets.split(';')

    for input_path in input_paths:
        all_preds = []
        all_true = []
        counter = 0

        num_tokens = 0
        num_kw = 0

        print("Folder: ", input_path)
        df = file_to_df(input_path, True)
        print("Num all docs: ", df.shape)


        for idx, row in df.iterrows():
            text = row['text'].lower()
            tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
            if args.lang == 'english':
                text_stem = " ".join([stemmer.stem(word) for word in tokens])
            elif args.lang == 'estonian' or args.lang == 'croatian':
                text_stem = " ".join([stemmer.lemmatize(word) for word in tokens])
            kw = row['keyword'].split(';')
            counter += 1
            if counter % 1000 == 0:
                print("Processing doc: ", counter)

            num_tokens += len(tokens)

            with open('temp.txt', 'w', encoding='utf8') as f:
                f.write(text)
            extractor = pke.unsupervised.TextRank()
            extractor.load_document(input='temp.txt')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            preds = extractor.get_n_best(n=numOfKeywords)
            preds = [x[0].lower() for x in preds]

            '''custom_kw_extractor = yake.KeywordExtractor(top=numOfKeywords)
            preds = custom_kw_extractor.extract_keywords(text)
            preds = [x[0].lower() for x in preds]'''


            true = [x.lower() for x in kw]
            #print("Text: ", text)
            #print("Preds: ", preds)
            #print("True: ", true)
            #print()
            #print()
            num_kw += len(true)
            true_in_text = []

            for t in true:
                if args.lang == 'english':
                    kw_stem = " ".join([stemmer.stem(word) for word in t.split()])
                elif args.lang == 'estonian' or args.lang == 'croatian':
                    kw_stem = " ".join([stemmer.lemmatize(word) for word in t.split()])
                #only add if keyword in text
                if kw_stem in text_stem:
                    true_in_text.append(t)

            all_preds.append(preds)
            all_true.append(true_in_text)

            #print(stemmed_preds)
            #print(stemmed_true)

        print('Num docs: ', counter)
        print('Num tokens: ', num_tokens)
        print('Num kw: ', num_kw)
        print('Avg. kw per doc: ', num_kw/counter)

        '''
        yake croatian
        P@5:  0.06153846153846154
        R@5:  0.14066671948351336
        F1@5:  0.0856
        
        P@10:  0.059072225484439224
        R@10:  0.25816198926885947
        F1@10:  0.0961
        
        P@k:  0.05793386709417244
        R@k:  0.05792408044316441
        F1@k:  0.0579
        
        P@M:  0.059072225484439224
        R@M:  0.25816198926885947
        F1@M:  0.0961
        
        yake estonian
        P@5:  0.022349570200573068
        R@5:  0.04265603247303047
        F1@5:  0.0293
        
        P@10:  0.028034987181420598
        R@10:  0.10477748413819193
        F1@10:  0.0442
        
        P@k:  0.02061056413718154
        R@k:  0.01966139879589637
        F1@k:  0.0201
        
        P@M:  0.028034987181420598
        R@M:  0.10477748413819193
        F1@M:  0.0442
        
        textrank estonian
        P@5:  0.00024380435329010206
        R@5:  0.0004272859799929624
        F1@5:  0.0003
        
        P@10:  0.0009441583902449534
        R@10:  0.004109407491250351
        F1@10:  0.0015
        
        P@k:  0.0001558337103503745
        R@k:  7.346998751010212e-05
        F1@k:  0.0001
        
        P@M:  0.0009441583902449534
        R@M:  0.004109407491250351
        F1@M:  0.0015
        textrank croatian
        P@5:  0.00015658641612840087
        R@5:  0.0002185685391792262
        F1@5:  0.0002
        
        P@10:  0.001990729089965731
        R@10:  0.006108734353009162
        F1@10:  0.003
        
        P@k:  3.914660403210022e-05
        R@k:  3.914660403210022e-05
        F1@k:  0.0
        
        P@M:  0.001990729089965731
        R@M:  0.006108734353009162
        F1@M:  0.003
-----------------------------------------------------------------------
        '''

        eval(all_preds, all_true, lang=args.lang)