import pandas as pd
import json
from eval_seq2seq import eval

def file_to_df(input_path):
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
            fulltext = line.get("fulltext") or ''
            text = text + ' ' + fulltext

            num_words += len(text.split())
            try:
                kw = line['keywords']
            except:
                kw = line['keyword']
            if isinstance(kw, list):
                kw = ";".join(kw)

            title = " ".join(title.split())
            abstract = " ".join(abstract.split())
            fulltext = " ".join(fulltext.split())

            all_docs.append([title, abstract, fulltext, kw])

    df = pd.DataFrame(all_docs)
    df.columns = ["title", "abstract", "fulltext", "keywords"]
    print(input_path, 'data size: ', df.shape)
    print('Avg words: ', num_words/df.shape[0])
    return df

def preprocess(kws):
    try:
        print(kws)
        kws = kws.split(';')
        print(kws)
        preprocessed_kws = []
        for kw in kws:
            if len(kw) > 0:
                preprocessed_kws.append(kw.lower())
        return ";".join(preprocessed_kws)
    except:
        return ''

if __name__ == '__main__':
    df = file_to_df('data/croatian/croatian_test.json')
    df_preds = pd.read_csv('predictions/croatian_5_lm+bpe+rnn_croatian_big.csv', sep=',', encoding='utf8')
    df_all = pd.concat([df, df_preds], axis=1)
    df_all = df_all.rename(columns={"True": "keywords_in_text", "Predicted": "predicted"})
    df = df.applymap(str)
    df_all['keywords_in_text'] = df_all['keywords_in_text'].map(lambda x: preprocess(x))
    df_all['keywords'] = df_all['keywords'].map(lambda x: preprocess(x))
    df_all['predicted'] = df_all['predicted'].map(lambda x: preprocess(x))
    df_all = df_all[['keywords', 'keywords_in_text', 'predicted', "title", "abstract"]]
    true = df_all['keywords_in_text'].tolist()
    true = [x.split(';') for x in true]
    predicted = df_all['predicted'].tolist()
    predicted = [x.split(';') for x in predicted]
    print(true[:500])
    p_5, r_5, f_5, p_10, r_10, f_10, p_k, r_k, f_k, p_M, r_M, f_M = eval(predicted, true, lang='croatian')
    df_all.to_csv("croatian_predictions.csv", sep=',', encoding="utf8", index=False)

    '''df = pd.read_csv('predictions/croatian_predictions_check.csv', sep=',', encoding='utf8')
    df = df.applymap(str)
    predicted = df['predicted'].tolist()
    predicted = [x.split(';') for x in predicted]
    true = df['keywords_in_text'].tolist()
    true = [x.split(';') for x in true]
    p_5, r_5, f_5, p_10, r_10, f_10, p_k, r_k, f_k, p_M, r_M, f_M = eval(predicted, true, lang='croatian')'''

