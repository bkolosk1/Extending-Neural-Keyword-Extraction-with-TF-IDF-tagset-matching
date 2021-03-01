import sentencepiece as spm
import os
from nltk import sent_tokenize
import json
import pandas as pd

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



def train_bpe_model(input, output):
    if input is not None:
        df = file_to_df(os.path.join('../data/' + input), classification=False)
        with open('../data/' + output + '.txt', 'w', encoding='utf8') as f:
            for idx, line in df.iterrows():
                text = line['text']
                sents = sent_tokenize(text)
                for sent in sents:
                    f.write(sent.lower().strip() + '\n')

    assert not os.path.exists(output + '.model')

    spm.SentencePieceTrainer.Train('--input=../data/' + output + '.txt --model_prefix=' + output + ' --vocab_size=32000 --character_coverage=1.0')

    sp = spm.SentencePieceProcessor()
    sp.Load(output + ".model")


    '''print(sp.EncodeAsIds("This is a test"))
    a = sp.EncodeAsPieces("who the hell decided to do this entropy is awsome.")
    print(a)
    a = sp.EncodeAsPieces("")
    print(a)
    print(sp.DecodePieces(a))'''


if __name__ == '__main__':
    input = 'estonian_big.json'
    output = 'bpe_estonian_big'
    train_bpe_model(input, output)
