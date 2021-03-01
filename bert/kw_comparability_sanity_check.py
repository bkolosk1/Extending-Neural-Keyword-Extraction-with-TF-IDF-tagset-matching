import pandas as pd
with open('seq2seq_semeval.txt', 'r', encoding='utf8') as f:
    gold = []
    gs = False
    kws = []
    for line in f:
        if line.startswith('[GROUND-TRUTH]'):
            gs = True
        elif line.strip().startswith('[PREDICTION]') :

            gs=False
            gold.append(kws)
            kws=[]
        elif gs:
            if line.strip().startswith('['):
                kw = line.strip()
                kw = kw.replace('[', '')
                kw = kw.replace(']', '')
                kw = " ".join(kw.strip().split())
                kws.append(kw)






df = pd.read_csv('predictions/semeval.csv', encoding='utf8', sep=',')

gold_me = []
for idx, row in df.iterrows():
    gs = row['True']
    gold_me.append(str(gs))
    #print(gs)

counter = 0

for kws in gold:
    kws = ";".join(kws)
    found = False
    for kws_me in gold_me:
        kws_me = kws_me.replace('-', ' ')

        if kws and kws_me and kws == kws_me:
            counter += 1
            #print(kws)
            #print(kws_me)
            #print()
            found = True
    if not found:
        print(kws)

print(counter)









