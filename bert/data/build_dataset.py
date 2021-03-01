import os
import json


from nltk.stem.porter import *

def stats():
    folders = ['kptimes', 'jptimes', 'duc']
    stemmer = PorterStemmer()
    all_words = 0
    train_kw = set()


    for fold in folders:
        train_path = os.path.join(fold, fold + '_train.json')
        valid_path = os.path.join(fold, fold + '_valid.json')
        test_path = os.path.join(fold, fold + '_test.json')

        all_paths = [train_path, test_path, valid_path]

        for path in all_paths:
            doc_counter = 0
            kw_counter = 0
            pres_kw_counter = 0
            num_words = 0
            num_words_in_kw = 0
            distinct_kw = set()
            num_in_train = 0
            if path == 'kptimes/kptimes_train.json':
                train_kw = set()

            empty = 0


            if os.path.exists(path):

                with open(path, 'r', encoding='utf8') as f:
                    for line in f:
                        doc_counter += 1

                        line = json.loads(line)
                        title = line.get('title') or ''
                        abstract = line.get('abstract') or ''
                        text = title + ' ' + abstract
                        num_words += len(text.split())
                        text = " ".join([stemmer.stem(word) for word in text.split()])

                        try:
                            kw = line['keywords']
                        except:
                            kw = line['keyword']
                        if not isinstance(kw, list):
                            kw = kw.split(';')
                        stemmed_kw = []
                        for k in kw:
                            k = [stemmer.stem(word) for word in k.split()]
                            stemmed_kw.append(" ".join(k))

                        kw = stemmed_kw
                        #print(kw)

                        kw_counter += len(kw)
                        kw_doc_pres_counter = 0
                        for k in kw:
                            if k.lower() in text.lower():
                                kw_doc_pres_counter += 1

                                pres_kw_counter += 1
                                num_words_in_kw += len(k.split())
                                distinct_kw.add(k)
                                if path in ['kp20k/kp20k_train.json', 'kptimes/kptimes_train.json']:
                                    train_kw.add(k.lower())
                                else:
                                    if k.lower() in train_kw:
                                        num_in_train += 1


                        if kw_doc_pres_counter == 0:
                            empty += 1

                    #print(fold)

                    if 'train' in path:
                        ffold = fold + '-train'
                    elif 'test' in path:
                        ffold = fold + '-test'
                    elif 'valid' in path:
                        ffold = fold + '-valid'

                    num_docs = doc_counter
                    avg_kw = "{0:.2f}".format(kw_counter/doc_counter)
                    avg_doc_len = "{0:.2f}".format(num_words / doc_counter)
                    avg_pres_kw_len = "{0:.2f}".format(num_words_in_kw / pres_kw_counter)
                    pres_kw_perc =  "{0:.2f}".format(pres_kw_counter/kw_counter*100)
                    avg_num_kw_present =  "{0:.2f}".format((kw_counter/doc_counter) * (pres_kw_counter/kw_counter))
                    avg_distinct = "{0:.2f}".format(len(distinct_kw)/doc_counter)
                    perc_in_train = "{0:.2f}".format(num_in_train/pres_kw_counter)

                    print(" & ".join([ffold, str(num_docs), avg_doc_len, str(avg_kw), str(pres_kw_perc), str(avg_num_kw_present)]) + ' \\\\')
                    print("Distinct: ", len(distinct_kw), avg_distinct)
                    print('% in train: ', perc_in_train)
                    all_words += num_words
                    print('Docs with no present keywords: ', empty)

                    #print('% pres: ', pres_kw_counter/kw_counter)
                    #print('Num docs: ', doc_counter)
                    #print('avg kw: ', kw_counter/doc_counter)
                    #print('avg words: ', num_words / doc_counter)
                    #print('avg kw length: ', num_words_in_kw / pres_kw_counter)
                    #print('avg num. kw present: ', (kw_counter/doc_counter) * (pres_kw_counter/kw_counter))

    print("Num all words: ", all_words)



def build_all_dataset():
    folders = ['inspec', 'kp20k', 'krapivin', 'nus', 'semeval']

    output_all_data = open('data_all.json', 'w', encoding='utf8')

    for fold in folders:
        test_path = os.path.join(fold, fold + '_test.json')
        valid_path = os.path.join(fold, fold + '_valid.json')

        counter = 0

        with open(test_path, 'r', encoding='utf8') as f:
            for line in f:
                output_all_data.write(line)
                counter += 1

        with open(valid_path, 'r', encoding='utf8') as f:
            for line in f:
                output_all_data.write(line)
                counter += 1

        print(fold, counter)

    folders = ['duc', 'kptimes']

    for fold in folders:
        test_path = os.path.join(fold, fold + '_test.json')
        valid_path = os.path.join(fold, fold + '_valid.json')

        counter = 0

        with open(test_path, 'r', encoding='utf8') as f:
            for line in f:
                output_all_data.write(line)
                counter += 1

        if fold == 'kptimes':
            with open(valid_path, 'r', encoding='utf8') as f:
                for line in f:
                    output_all_data.write(line)
                    counter += 1

        print(fold, counter)

    output_all_data.close()



def build_science_dataset():
    folders = ['inspec', 'kp20k', 'krapivin', 'nus', 'semeval']
    output_all_data = open('data_science_big.json', 'w', encoding='utf8')

    for fold in folders:
        test_path = os.path.join(fold, fold + '_test.json')
        valid_path = os.path.join(fold, fold + '_valid.json')

        counter = 0

        with open(test_path, 'r', encoding='utf8') as f:
            for line in f:
                output_all_data.write(line)
                counter += 1

        if fold != 'nus':
            with open(valid_path, 'r', encoding='utf8') as f:
                for line in f:
                    output_all_data.write(line)
                    counter += 1

        if fold == 'kp20k':
            train_path = os.path.join(fold, fold + '_train.json')
            with open(train_path, 'r', encoding='utf8') as f:
                for line in f:
                    #if counter < 80000:
                    output_all_data.write(line)
                    #else:
                    #    break
                    counter += 1

        print(fold, counter)



    output_all_data.close()


def build_news_dataset():
    folders = ['duc', 'kptimes', 'jptimes']

    output_all_data = open('data_news_big.json', 'w', encoding='utf8')

    for fold in folders:
        test_path = os.path.join(fold, fold + '_test.json')
        valid_path = os.path.join(fold, fold + '_valid.json')

        counter = 0

        with open(test_path, 'r', encoding='utf8') as f:
            for line in f:
                output_all_data.write(line)
                counter += 1

        if fold != 'duc':
            with open(valid_path, 'r', encoding='utf8') as f:
                for line in f:
                    output_all_data.write(line)
                    counter += 1

        if fold == 'kptimes':
            train_path = os.path.join(fold, fold + '_train.json')
            with open(train_path, 'r', encoding='utf8') as f:
                for line in f:
                    #if counter < 80000:
                    output_all_data.write(line)
                    #else:
                    #    break
                    counter += 1

        print(fold, counter)

    output_all_data.close()


def split_kptimes():
    folders = ['kptimes']

    output_all_data = open('kptimes_test.json', 'w', encoding='utf8')

    for fold in folders:
        test_path = os.path.join(fold, fold + '_test.json')

        counter = 0

        with open(test_path, 'r', encoding='utf8') as f:
            for line in f:
                print(counter)
                if counter >= 10000:
                    output_all_data.write(line)
                counter += 1



        print(fold, counter)

    output_all_data.close()



if __name__ == '__main__':
    #build_science_dataset()
    #build_news_dataset()
    #build_all_dataset()
    stats()
    #split_kptimes()






