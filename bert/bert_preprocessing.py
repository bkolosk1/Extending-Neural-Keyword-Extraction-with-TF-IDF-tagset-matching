import torch
from nltk import sent_tokenize, word_tokenize
from collections import defaultdict
import json
import pandas as pd
import pickle
from nltk.tag.perceptron import PerceptronTagger
from nltk.stem.porter import *
from transformers import BertTokenizer, GPT2Tokenizer
from lemmagen3 import Lemmatizer
import LatvianStemmer
#unk_token = '<|endoftext|>'
#eos_token = '<|endoftext|>'
#pad_token = '<pad>'
eos_token = '<eos>'
unk_token = '<unk>'
pad_token = '<pad>'



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



class Dictionary(object):
    def __init__(self, max_vocab_size):
        self.word2idx = {pad_token:0}
        self.idx2word = {0:pad_token}
        self.counter = {}
        self.total = 0
        self.max_vocab_size = max_vocab_size


    def add_word(self, word, idx):
        self.counter.setdefault(word, 0)
        self.counter[word] += 1
        self.total += 1
        self.idx2word[idx] = word
        self.word2idx[word] = idx

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, df_train, df_valid, df_test, args, unk=True):

        self.dictionary = Dictionary(max_vocab_size=args.max_vocab_size)
        self.unk = unk
        self.classification = args.classification
        self.transfer_learning = args.transfer_learning
        self.max_length = args.n_ctx
        self.pos = args.POS_tags
        self.pos_tags = set()
        self.bpe = args.bpe
        self.lang = args.lang
        if self.bpe:
            self.sp = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
            #self.sp = GPT2Tokenizer.from_pretrained("gpt2", add_prefix_space=True)

        if self.pos:
            self.tagger = PerceptronTagger()

        self.tokenize_df(df_train)
        self.tokenize_df(df_valid)
        self.tokenize_df(df_test)

        with open(args.dict_path, 'wb') as file:
            pickle.dump(self.dictionary, file)

        if not self.classification:
            if self.pos:
                self.train, self.train_pos = self.tokenize_(df_train)
                self.valid, self.valid_pos = self.tokenize_(df_valid)
                self.test, self.test_pos = self.tokenize_(df_test)
            else:
                self.train = self.tokenize_(df_train)
                self.valid = self.tokenize_(df_valid)
                self.test = self.tokenize_(df_test)
        else:
            if self.pos:
                self.train, self.train_pos, self.train_target, self.train_keywords, self.train_stemmed_string = self.tokenize_doc(df_train, max_length=self.max_length)
                self.valid, self.valid_pos, self.valid_target, self.valid_keywords, self.valid_stemmed_string = self.tokenize_doc(df_valid, max_length=self.max_length, valid=False)
                self.test, self.test_pos, self.test_target, self.test_keywords, self.test_stemmed_string = self.tokenize_doc(df_test, max_length=self.max_length)
            else:
                self.train, self.train_target, self.train_keywords, self.train_stemmed_string = self.tokenize_doc(df_train, max_length=self.max_length)
                self.valid, self.valid_target, self.valid_keywords, self.valid_stemmed_string = self.tokenize_doc(df_valid, max_length=self.max_length, valid=False)
                self.test, self.test_target, self.test_keywords, self.test_stemmed_string = self.tokenize_doc(df_test, max_length=self.max_length)

    def preprocess_line(self, line, pos):
        words = []
        if pos:
            pos_tags = []

        text = line['text']
        #print(self.sp.tokenize(text))
        text = text.replace('-', ' ')
        text = text.replace('/', ' ')
        text = text.replace('∗', ' ')
        for sent in sent_tokenize(text):
            sent = word_tokenize(sent)
            if self.bpe:
                bpe_sent = []
                for w in sent:
                    w = w.lower()
                    bpe_word = self.sp.tokenize(w)
                    bpe_sent.append(bpe_word)
                    words.extend(bpe_word)
                words.append(eos_token)
            else:
                words.extend([w.lower() for w in sent] + [eos_token])
            if pos:
                pos_sent = [x[1] for x in self.tagger.tag(sent)]
                if self.bpe:
                    for pos, bpe_token in zip(pos_sent, bpe_sent):
                        for i in range(len(bpe_token)):
                            pos_tags.append(pos)
                    pos_tags.append(eos_token)
                else:
                    pos_tags.extend(pos_sent + [eos_token])

        if pos:
            return words, pos_tags
        #print(words)
        return words


    def tokenize_df(self, df):

        counter = 0
        tokens = 0

        for idx, line in df.iterrows():
            counter += 1

            if self.pos:
                words, pos_tags = self.preprocess_line(line, self.pos)
            else:
                words = self.preprocess_line(line, self.pos)

            for word in words:
                idx = self.sp.convert_tokens_to_ids(word)
                self.dictionary.add_word(word, idx)
            if self.pos:
                for pt in pos_tags:

                    # only count special POS tags symbols that are out of vocabulary
                    if len(pt) > 1 and pt not in ['``', "''"]:
                        self.pos_tags.add(pt)

            tokens += len(words)

            if counter % 1000 == 0:
                print('Processing doc: ', counter)
        print("Num tokens: ", tokens)
        return


    def tokenize_(self, df):

        counter = 0
        tokens = 0
        for idx, line in df.iterrows():
            words = self.preprocess_line(line, pos=False)
            tokens += len(words)


        ids = torch.LongTensor(tokens)
        if self.pos:
            ids_pos = torch.LongTensor(tokens)

        token = 0
        if self.pos:
            token_pos = 0
        for idx, line in df.iterrows():
            counter += 1
            if self.pos:
                words, pos_tags = self.preprocess_line(line, self.pos)
            else:
                words = self.preprocess_line(line, self.pos)

            for word in words:
                if word in self.dictionary.word2idx:
                    idx = self.dictionary.word2idx[word]
                else:
                    idx = self.dictionary.word2idx[unk_token]
                ids[token] = idx
                token += 1

            if self.pos:
                for pt in pos_tags:
                    if pt in self.dictionary.word2idx:
                        idx = self.dictionary.word2idx[pt]
                    else:
                        idx = self.dictionary.word2idx[unk_token]
                    ids_pos[token_pos] = idx
                    token_pos += 1
            if counter % 1000 == 0:
                print('Processing doc: ', counter)
        if not self.pos:
            return ids
        return ids, ids_pos



    def tokenize_doc(self, df, max_length, valid=False):
        if self.lang=='english':
            stemmer = PorterStemmer()
        if self.lang == 'latvian':
            stemmer = LatvianStemmer
        if self.lang == 'croatian':
            stemmer = Lemmatizer('hr')
        if  self.lang == 'russian':
            stemmer = Lemmatizer('ru')
        if self.lang=='estonian':
            stemmer = Lemmatizer('et')
    
        """
        if self.lang == 'english':
            stemmer = PorterStemmer()
        elif self.lang == 'estonian':
            stemmer = Lemmatizer('et')
        elif self.lang == 'croatian':
            stemmer = Lemmatizer('hr')
        """
        stemmed_string = ""

        docs = []
        for idx, line in df.iterrows():
            if self.pos:
                words, pos_tags = self.preprocess_line(line, self.pos)
            else:
                words = self.preprocess_line(line, self.pos)
            if self.lang == 'english' or self.lang=='latvian':
                stems = " ".join([stemmer.stem(w.lower()) for w in words])
            elif self.lang == 'estonian' or self.lang=='croatian' or self.lang == 'russian':
                stems = " ".join([stemmer.lemmatize(w.lower()) for w in words])
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
                    kw = self.sp.tokenize(kw)
                tokenized_keywords.append(kw)

            if self.pos:
                docs.append([words, pos_tags, tokenized_keywords])
            else:
                docs.append([words, tokenized_keywords])

        x = torch.zeros([len(docs), max_length], dtype=torch.long)
        y = torch.zeros([len(docs), max_length], dtype=torch.long)
        if self.pos:
            x_pos = torch.zeros([len(docs), max_length], dtype=torch.long)

        all_keywords = {}
        not_in_text = defaultdict(int)
        present_kw = 0
        all_kw = 0
        copies = 0
        max_lkw = 4

        for i, doc in enumerate(docs):
            if self.pos:
                words, pos_tags, kws = doc
            else:
                words, kws = doc
            if valid:
                print(kws)
                print(words)
                if self.pos:
                    print(pos_tags)

            length = len(words)

            kw_in_paper = []
            stemmed_kw_in_paper = []


            for j, word in enumerate(words):
                if word in self.dictionary.word2idx:
                    idx = self.dictionary.word2idx[word]

                    for kw in kws:
                        lkw = len(kw)

                        is_keyword = False
                        if j + lkw < length:
                            for k in range(lkw):
                                w = words[j + k]

                                if self.lang == 'english' or self.lang=='latvian':
                                    if stemmer.stem(w.lower()) != stemmer.stem(kw[k].lower()):
                                        break
                                elif self.lang == 'estonian' or self.lang=='croatian' or self.lang == 'russian':
                                    if stemmer.lemmatize(w.lower()) != stemmer.lemmatize(kw[k].lower()):
                                        break

                            else:
                                is_keyword = True
                        if is_keyword:

                            for k in range(lkw):
                                if j + k < max_length:
                                    y[i][j + k] = lkw + 1 if lkw <= max_lkw else max_lkw + 1

                            kw_in_paper.append(" ".join(kw))

                            if self.lang == 'english' or self.lang=='latvian':
                                stemmed_kw = " ".join([stemmer.stem(w.lower()) for w in kw])
                            elif self.lang == 'estonian' or self.lang=='croatian' or self.lang == 'russian':
                                stemmed_kw = " ".join([stemmer.lemmatize(w.lower()) for w in kw])

                            stemmed_kw_in_paper.append(stemmed_kw)

                else:
                    idx = self.dictionary.word2idx[unk_token]
                if j < max_length:
                    x[i][j] = idx
                    if y[i][j] == 0:
                        y[i][j] = 1

            if self.pos:
                for j, pt in enumerate(pos_tags):
                    if pt in self.dictionary.word2idx:
                        idx = self.dictionary.word2idx[pt]
                    else:
                        idx = self.dictionary.word2idx[unk_token]
                    if j < max_length:
                        x_pos[i][j] = idx

            if valid:
                print(x[i])
                print(y[i])
                if self.pos:
                    print(x_pos[i])

            key = "".join([str(idx) for idx in x[i].numpy()])

            #remove keywords that don't appear
            num_all_kw = len(kws)
            not_kws = [" ".join(x) for x in kws if " ".join(x) not in kw_in_paper]
            kws = [x for x in kws if " ".join(x) in kw_in_paper]

            for k in not_kws:
                not_in_text[k] += 1

            all_kw += num_all_kw
            present_kw += len(kws)



            if key not in all_keywords:
                all_keywords[key] = kws
            else:
                copies += 1
                #print('TWO identical keys!')
                #print(key)
                #print([self.dictionary.idx2word[idx] for idx in x[i].numpy()])

        print('Num all keywords: ', all_kw)
        print('Percentage of kw. present: ', present_kw / all_kw)

        print('Num identical keys: ', copies)

        l = sorted(not_in_text.items(), key=lambda x: x[1], reverse=True)
        print('Num. keywords that do not appear inside text: ', len(l))
        print('Most common out of text kw: ', l[:100])
        print('Max kw length: ', max_lkw)

        print('X Y size: ', x.size(), y.size())
        if self.pos:
            return x, x_pos, y, all_keywords, stemmed_string
        return x, y, all_keywords, stemmed_string


def batchify(data, bsz, n_ctx):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).contiguous()
    data = data[:, :data.size(1) - data.size(1) % n_ctx]
    data = data
    return data


def get_batch(source, i, config, word2idx, masked_indices=None):

    encoder = source[:,i:i+config.n_ctx]

    if config.masked_lm:
        encoder = encoder.clone()

        if masked_indices is None:
             masked_indices = torch.bernoulli(torch.full(encoder.shape, 0.15)).bool()
        target = encoder.clone()
        target[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(encoder.shape, 0.8)).bool() & masked_indices
        encoder[indices_replaced] = word2idx['<mask>']

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(encoder.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(word2idx), encoder.shape, dtype=torch.long)
        encoder[indices_random] = random_words[indices_random]
        return encoder.cuda(), target.cuda(), masked_indices

    else:
        target = source[:, i + 1:i + 1 + config.n_ctx]
        return encoder.cuda(), target.cuda(), None




def batchify_docs(data, targets, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    doc_length = data.size(1)
    target_length = targets.size(1)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).

    data = data.narrow(0, 0, nbatch * bsz)
    targets = targets.narrow(0, 0, nbatch * bsz)


    #Evenly divide the data across the bsz batches.
    data = data.view(-1, bsz, doc_length).contiguous()
    targets = targets.view(-1, bsz, target_length).contiguous()

    #print(data.size(), targets.size())
    return data, targets


def get_batch_docs(data, targets, i):
    return data[i, :, :].cuda(), targets[i, :, :].cuda()




