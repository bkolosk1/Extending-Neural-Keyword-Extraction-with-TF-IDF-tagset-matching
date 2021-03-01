import re

def parse_to_latex():
    configs = ['nolm', 'lm', 'maskedlm', 'lm+bp', 'lm+pos', 'lm+rnn', 'lm+bpe+rnn', 'lm+bpe+crf']
    datasets = ['kp20k', 'inspec', 'krapivin', 'nus', 'semeval', 'kptimes', 'jptimes', 'duc']
    config_dict = {}

    with open('class_results-FINAL.txt', 'r', encoding='utf8') as file:
        for line in file.readlines():
            if line.startswith('Classification'):
                config = line.split('_')[-3]
                print(config)
            if line.startswith('Dataset:'):
                dataset = line.split()[-1]
                print(dataset)
            if line.startswith('Precision') and not line.startswith('Precision@M:') and not line.startswith(('Precision@k')):
                measure = line.split()[-2][:-1]
                score = line.split()[-1]
                print(measure, score)

                if config not in config_dict:
                    config_dict[config] = {}
                    config_dict[config][dataset] = [(measure, score)]
                else:
                    if dataset in config_dict[config]:
                        config_dict[config][dataset].append((measure, score))
                    else:
                        config_dict[config][dataset] = [(measure, score)]

    lines = []
    average5 = []
    average10 = []
    for config in configs:
        sum5 = 0
        sum10 = 0
        column = []
        for dataset in datasets:
            column.append(dataset)
            for e in config_dict[config][dataset]:
                column.append((e[0], e[1]))
                if e[0].endswith('10'):
                    sum10 += float(e[1])
                if e[0].endswith('5'):
                    sum5 += float(e[1])
        sum10 = sum10/len(datasets)
        sum5 = sum5/len(datasets)
        average5.append(sum5)
        average10.append(sum10)
        lines.append(column)

    print(lines)

    print("& " + " & ".join(configs) + '\\\\\\hline')

    for i in range(len(lines[0])):
        if i % 3 == 0:
            dataset = lines[0][i]
            #print(dataset)
            print('& \\multicolumn{8}{c}{\\textbf{' + dataset + '}}\\\\\\hline')
        else:
            #print(lines[0])
            line = lines[0][i][0] + " & " + " & ".join([x[i][1] for x in lines]) + '\\\\'
            print(line)
    print('& \\multicolumn{7}{c}{\\textbf{Average}}\\\\\\hline')
    print("F@5 & " + " & ".join(["{:.4f}".format(x) for x in average5]) + '\\\\')
    print("F@10 & " + " & ".join(["{:.4f}".format(x) for x in average10]) + '\\\\')

#parse_to_latex()

def get_averages():
    results ='''
    & \multicolumn{9}{c}{\textbf{KP20k}} \\\hline
    F1@5 & 0.072 & 0.181 & 0.141* & 0.177* & 0.046 & 0.005 & 0.317 & \textbf{0.348} & 0.252* & 0.339* & 0.342*\\
    F1@10 & 0.094 & 0.151 & 0.146* & 0.160* & 0.044 & 0.005 & 0.273 & 0.298 & 0.256* & 0.342* & \textbf{0.346*}\\
    \hline
    & \multicolumn{9}{c}{\textbf{Inspec}} \\\hline
    F1@5 & 0.160 & 0.286 & 0.204* & 0.101* & 0.022 & 0.035 & 0.244 & 0.276 & 0.293* & \textbf{0.467*} & 0.447*\\
    F1@10 & 0.244 & 0.339 & 0.223* & 0.108* & 0.022 & 0.046 & 0.289 & 0.333 & 0.335* & \textbf{0.525*} & \textbf{0.525*}\\
    \hline
    & \multicolumn{9}{c}{\textbf{Krapivin}} \\\hline
    F1@5 & 0.067 & 0.185 & 0.215* & 0.127* & 0.018 & 0.005 & 0.305 & \textbf{0.325} & 0.210* & 0.280* & 0.301*\\
    F1@10 & 0.093 & 0.160 & 0.196* & 0.106* & 0.017 & 0.007 & 0.266 & 0.285 & 0.214* & 0.283* & \textbf{0.307*}\\
    \hline
    & \multicolumn{9}{c}{\textbf{NUS}} \\\hline
    F1@5 & 0.112 & 0.230 & 0.159* & 0.224* & 0.073 & 0.004 & 0.376 & \textbf{0.374} & 0.274* & 0.311* & 0.350*\\
    F1@10 & 0.140 & 0.216 & 0.196* & 0.193* & 0.071 & 0.006 & 0.352 & 0.366 & 0.305* & 0.332* & \textbf{0.369*}\\
    \hline
    & \multicolumn{9}{c}{\textbf{SemEval}} \\\hline
    F1@5 & 0.088 & 0.217 & 0.151* & 0.167* & 0.068 & 0.011 & 0.318 & \textbf{0.327} & 0.261* & 0.214 & 0.291*\\
    F1@10 & 0.147 & 0.226 & 0.212* & 0.159* & 0.065 & 0.014 & 0.318 & 0.352 & 0.295* & 0.232 & \textbf{0.355*}\\
    \hline\hline
    & \multicolumn{9}{c}{\textbf{KPTimes}} \\\hline
    F1@5 & 0.179* & 0.022* & 0.105* & 0.168* & * & * & 0.406* & 0.424* & 0.353* & 0.439* & \textbf{0.469*}\\
    F1@10 & 0.151* & 0.030* & 0.118* & 0.139* & * & * & 0.393 & 0.424* & 0.354* & 0.440* & \textbf{0.469*}\\\hline
    & \multicolumn{9}{c}{\textbf{JPTimes}} \\\hline
    F1@5 & 0.266* & 0.012* & 0.109* & 0.225* & * & * & 0.256* & 0.238* & 0.258* & \textbf{0.344*} & 0.337*\\
    F1@10 & 0.229* & 0.026* & 0.135* & 0.185* & * & * & 0.246 & 0.238* & 0.267* & 0.346* & \textbf{0.360*}\\\hline
    & \multicolumn{9}{c}{\textbf{DUC}} \\\hline
    F1@5 & 0.098* & 0.120* & 0.106* & 0.189* & * & * & 0.083 & 0.063* & 0.247* & 0.281* & \textbf{0.312*}\\
    F1@10 & 0.120* & 0.181* & 0.132* & 0.172* & * & * & 0.105 & 0.063* & 0.277* & 0.321* & \textbf{0.355*}\\\hline
    '''
    f5s = [[], [], [], [], [], [], [], [], [], [], []]
    f10s = [[], [], [], [], [], [], [], [], [], [], []]

    for line in results.split('\n'):
        line = line.strip()
        print(line)
        if line.startswith('F1@5'):
            line = line.split('&')
            line = line[1:]
            for idx, score in enumerate(line):
                score = score.strip()
                score = re.findall(r'\d+', score)
                if len(score) > 0:
                    f5s[idx].append((float(".".join(score))))
                else:
                    f5s[idx].append(0)
        elif line.startswith('F1@10'):
            line = line.split('&')
            line = line[1:]
            for idx, score in enumerate(line):
                score = score.strip()
                #print(score)
                score = re.findall(r'\d+', score)
                #print(score)
                if len(score) > 0:
                    f10s[idx].append((float(".".join(score))))
                else:
                    f10s[idx].append(0)
    print(f5s)
    print(f10s)

    f5s = " & ".join(['F1@5'] + ["{:.4f}".format(sum(x)/len(x)) for x in f5s])
    f10s = " & ".join(['F1@10'] +  ["{:.4f}".format(sum(x)/len(x)) for x in f10s])

    print(f5s)
    print(f10s)


#get_averages()

def revert():
    results = '''& TfIdf & TextRank & YAKE & RaKUn & Key2Vec & EmbedRank & KEA & Maui & SemiSupervised & CopyRNN & CatSeqD & CorrRNN & GPT-2 & \begin{tabular}[x]{@{}c@{}}GPT-2 + \\ BiLSTM-CRF\end{tabular} & TNT-KID \\\hline
    & \multicolumn{9}{c}{\textbf{KP20k}} \\\hline
    F1@5 & 0.072 & 0.181 & 0.141* & 0.177* & 0.080* & 0.135* & 0.046 & 0.005 & 0.308 & 0.317 & \textbf{0.348} & / & 0.252* & 0.343* & 0.338*\\
    F1@10 & 0.094 & 0.151 & 0.146* & 0.160* & 0.090* & 0.134* & 0.044 & 0.005 & 0.245 & 0.273 & 0.298 & / & 0.256* & \textbf{0.347*} & 0.342*\\
    \hline
    & \multicolumn{9}{c}{\textbf{Inspec}} \\\hline
    F1@5 & 0.160 & 0.286 & 0.204* & 0.101* & 0.121* & 0.345* & 0.022 & 0.035 & 0.326  & 0.244 & 0.276 & / & 0.293* & \textbf{0.468*} & 0.456*\\
    F1@10 & 0.244 & 0.339 & 0.223* & 0.108* & 0.181* & 0.394* & 0.022 & 0.046 & 0.334  & 0.289 & 0.333 & / &  0.335* & \textbf{0.535*} & 0.534*\\
    \hline
    & \multicolumn{9}{c}{\textbf{Krapivin}} \\\hline
    F1@5 & 0.067 & 0.185 & 0.215* & 0.127* & 0.068* & 0.149* & 0.018 & 0.005 & 0.296 & 0.305 & \textbf{0.325} & 0.318 & 0.210* & 0.302* & 0.313*\\
    F1@10 & 0.093 & 0.160 & 0.196* & 0.106* & 0.082* & 0.158* & 0.017 & 0.007 & 0.240 & 0.266 & 0.285 & 0.278 & 0.214* & 0.302* & \textbf{0.318*}\\
    \hline
    & \multicolumn{9}{c}{\textbf{NUS}} \\\hline
    F1@5 & 0.112 & 0.230 & 0.159* & 0.224* & 0.109* & 0.173* & 0.073 & 0.004 & 0.356 & 0.376 & \textbf{0.374} & 0.361 & 0.274* & 0.315* & 0.345*\\
    F1@10 & 0.140 & 0.216 & 0.196* & 0.193* & 0.121* & 0.190* & 0.071 & 0.006 & 0.320 & 0.352 & 0.366 & 0.335 & 0.305* & 0.333* & \textbf{0.357*}\\
    \hline
    & \multicolumn{9}{c}{\textbf{SemEval}} \\\hline
    F1@5 & 0.088 & 0.217 & 0.151* & 0.167* & 0.081* & 0.189* & 0.068 & 0.011 & 0.322  & 0.318 & \textbf{0.327} & 0.320 & 0.261* & 0.262 & 0.294*\\
    F1@10 & 0.147 & 0.226 & 0.212* & 0.159* & 0.126* & 0.217* & 0.065 & 0.014 & 0.294 & 0.318 & \textbf{0.352} & 0.320 & 0.295* & 0.273 & 0.334*\\
    \hline\hline
    & \multicolumn{9}{c}{\textbf{KPTimes}} \\\hline
    F1@5 & 0.179* & 0.022* & 0.105* & 0.168* & 0.126* & 0.063* & * & * & /  & 0.406* & 0.424* & / & 0.353* & \textbf{0.497*} & \textbf0.488*\\
    F1@10 & 0.151* & 0.030* & 0.118* & 0.139* & 0.116* & 0.057* & * & * & /  & 0.393 & 0.424* & / & 0.354* & \textbf{0.497*} & 0.486*\\\hline
    & \multicolumn{9}{c}{\textbf{JPTimes}} \\\hline
    F1@5 & 0.266* & 0.012* & 0.109* & 0.225* & 0.158* & 0.081* & * & * & /  & 0.256* & 0.238* & / & 0.258* & 0.375* & \textbf{0.385*}\\
    F1@10 & 0.229* & 0.026* & 0.135* & 0.185* & 0.145* & 0.074* & * & * & /  & 0.246 & 0.238* & / & 0.267* & 0.380* & \textbf{0.385*}\\\hline
    & \multicolumn{9}{c}{\textbf{DUC}} \\\hline
    F1@5 & 0.098* & 0.120* & 0.106* & 0.189* & 0.062* & 0.219* & * & * & /  & 0.083 & 0.063* & / & 0.247* & \textbf{0.334*} & 0.310*\\
    F1@10 & 0.120* & 0.181* & 0.132* & 0.172* & 0.078* & 0.246* & * & * & /  & 0.105 & 0.063* & / & 0.277* & 0.369* & \textbf{0.372*}\\\hline
    '''
    re.sub("[^0-9]", "", "sdkjh987978asd098as0980a98sd")
    alg2idx = {}
    datasets = {}
    order = []


    for line in results.split('\n'):
        line = line.strip()
        if line.startswith('& TfIdf'):
            algs = line.replace('\\hline' ,'').replace('\\', '').split('&')[1:]
            for idx, alg in enumerate(algs):
                alg2idx[alg] = idx
        elif line.startswith('& \multicolumn'):
            dataset = line.replace('\\', '').replace('& multicolumn{9}{c}{', '').replace('}} hline', '').replace('extbf{', '').strip()
            order.append(dataset)
            if dataset not in datasets:
                datasets[dataset] = [[],[]]

        elif line.startswith('F1@5'):
            line = line.replace('\\hline' ,'').replace('\\', '')
            line = line.split('&')[1:]
            #print(line)
            for score in line:
                datasets[dataset][0].append(score)
        elif line.startswith('F1@10'):
            line = line.replace('\\hline', '').replace('\\', '')
            line = line.split('&')[1:]
            #print(line)
            for score in line:
                datasets[dataset][1].append(score)


    print(" & " + " & ".join(order) + ' \\\\\\hline')
    for alg in algs:
        line_f5 = ['F1@5']
        line_f10 = ['F1@10']
        for dataset in order:
            f5_score = datasets[dataset][0][alg2idx[alg]]
            line_f5.append(f5_score.replace('\t', '\\t'))
            f10_score = datasets[dataset][1][alg2idx[alg]]
            line_f10.append(f10_score.replace('\t', '\\t'))
        print('& \\multicolumn{8}{c}{\\textbf{' + alg + '}} \\\\')
        print(" & ".join(line_f5) + ' \\\\')
        print(" & ".join(line_f10) + ' \\\\\\hline')

#revert()


def get_averages_reverted():
    results ='''
    & KP20k & Inspec & Krapivin & NUS & SemEval & KPTimes & JPTimes & DUC \\\hline
    & \multicolumn{8}{c}{\textbf{ TfIdf }} \\
    F1@5 &  0.072  &  0.160  &  0.067  &  0.112  &  0.088  &  0.179*  &  0.266*  &  0.098*  \\
    F1@10 &  0.094  &  0.244  &  0.093  &  0.140  &  0.147  &  0.151*  &  0.229*  &  0.120*  \\\hline
    & \multicolumn{8}{c}{\textbf{ TextRank }} \\
    F1@5 &  0.181  &  0.286  &  0.185  &  0.230  &  0.217  &  0.022*  &  0.012*  &  0.120*  \\
    F1@10 &  0.151  &  0.339  &  0.160  &  0.216  &  0.226  &  0.030*  &  0.026*  &  0.181*  \\\hline
    & \multicolumn{8}{c}{\textbf{ YAKE }} \\
    F1@5 &  0.141*  &  0.204*  &  0.215*  &  0.159*  &  0.151*  &  0.105*  &  0.109*  &  0.106*  \\
    F1@10 &  0.146*  &  0.223*  &  0.196*  &  0.196*  &  0.212*  &  0.118*  &  0.135*  &  0.132*  \\\hline
    & \multicolumn{8}{c}{\textbf{ RaKUn }} \\
    F1@5 &  0.177*  &  0.101*  &  0.127*  &  0.224*  &  0.167*  &  0.168*  &  0.225*  &  0.189*  \\
    F1@10 &  0.160*  &  0.108*  &  0.106*  &  0.193*  &  0.159*  &  0.139*  &  0.185*  &  0.172*  \\\hline
    & \multicolumn{8}{c}{\textbf{ Key2Vec }} \\
    F1@5 &  0.080*  &  0.121*  &  0.068*  &  0.109*  &  0.081*  &  0.126*  &  0.158*  &  0.062*  \\
    F1@10 &  0.090*  &  0.181*  &  0.082*  &  0.121*  &  0.126*  &  0.116*  &  0.145*  &  0.078*  \\\hline
    & \multicolumn{8}{c}{\textbf{ EmbedRank }} \\
    F1@5 &  0.135*  &  0.345*  &  0.149*  &  0.173*  &  0.189*  &  0.063*  &  0.081*  &  0.219*  \\
    F1@10 &  0.134*  &  0.394*  &  0.158*  &  0.190*  &  0.217*  &  0.057*  &  0.074*  &  0.246*  \\\hline
    & \multicolumn{8}{c}{\textbf{ KEA }} \\
    F1@5 &  0.046  &  0.022  &  0.018  &  0.073  &  0.068  &  *  &  *  &  *  \\
    F1@10 &  0.044  &  0.022  &  0.017  &  0.071  &  0.065  &  *  &  *  &  *  \\\hline
    & \multicolumn{8}{c}{\textbf{ Maui }} \\
    F1@5 &  0.005  &  0.035  &  0.005  &  0.004  &  0.011  &  *  &  *  &  *  \\
    F1@10 &  0.005  &  0.046  &  0.007  &  0.006  &  0.014  &  *  &  *  &  *  \\\hline
    & \multicolumn{8}{c}{\textbf{ SemiSupervised }} \\
    F1@5 &  0.308  &  0.326   &  0.296  &  0.356  &  0.322   &  /   &  /   &  /   \\
    F1@10 &  0.245  &  0.334   &  0.240  &  0.320  &  0.294  &  /   &  /   &  /   \\\hline
    & \multicolumn{8}{c}{\textbf{ CopyRNN }} \\
    F1@5 &  0.317  &  0.244  &  0.305  &  0.376  &  0.318  &  0.406*  &  0.256*  &  0.083  \\
    F1@10 &  0.273  &  0.289  &  0.266  &  0.352  &  0.318  &  0.393  &  0.246  &  0.105  \\\hline
    & \multicolumn{8}{c}{\textbf{ CatSeqD }} \\
    F1@5 &  \textbf{0.348}  &  0.276  &  \textbf{0.325}  &  \textbf{0.374}  &  \textbf{0.327}  &  0.424*  &  0.238*  &  0.063*  \\
    F1@10 &  0.298  &  0.333  &  0.285  &  0.366  &  \textbf{0.352}  &  0.424*  &  0.238*  &  0.063*  \\\hline
    & \multicolumn{8}{c}{\textbf{ CorrRNN }} \\
    F1@5 &  /  &  /  &  0.318  &  0.361  &  0.320  &  /  &  /  &  /  \\
    F1@10 &  /  &  /  &  0.278  &  0.335  &  0.320  &  /  &  /  &  /  \\\hline
    & \multicolumn{8}{c}{\textbf{ GPT-2 }} \\
    F1@5 &  0.252*  &  0.293*  &  0.210*  &  0.274*  &  0.261*  &  0.353*  &  0.258*  &  0.247*  \\
    F1@10 &  0.256*  &   0.335*  &  0.214*  &  0.305*  &  0.295*  &  0.354*  &  0.267*  &  0.277*  \\\hline
    & \multicolumn{8}{c}{\textbf{\begin{tabular}[x]{@{}c@{}}GPT-2 +  BiLSTM-CRF\end{tabular} }} \\
    F1@5 &  0.343*  &  \textbf{0.468*}  &  0.302*  &  0.315*  &  0.262  &  \textbf{0.497*}  &  0.375*  &  \textbf{0.334*}  \\
    F1@10 &  \textbf{0.347*}  &  \textbf{0.535*}  &  0.302*  &  0.333*  &  0.273  &  \textbf{0.497*}  &  0.380*  &  0.369*  \\\hline
    & \multicolumn{8}{c}{\textbf{ TNT-KID }} \\
    F1@5 &  0.338* &  0.456* &  0.313* &  0.345* &  0.294* &  \textbf0.488* &  \textbf{0.385*} &  0.310* \\
    F1@10 &  0.342* &  0.534* &  \textbf{0.318*} &  \textbf{0.357*} &  0.334* &  0.486* &  \textbf{0.385*} &  \textbf{0.372*} \\\hline
    '''

    for line in results.split('\n'):
        line = line.strip()
        if line.startswith('F1') :
            scores = line.split('&')
            scores = scores[1:]
            clean = []
            for idx, score in enumerate(scores):
                score = score.strip()
                score = re.findall(r'\d+', score)
                if len(score) > 0:
                    clean.append((float(".".join(score))))
                else:
                    clean.append(0)
            avg = sum(clean)/len(clean)
            line = line.replace('\\', '').replace('hline', '')
            line = line + " & " + "{:.3f}".format(avg)
            if line.startswith('F1@5'):
                line += ' \\\\'
            else:
                line += ' \\\\\\hline'
            print(line.replace('\t', '\\t'))


        else:
            print(line.replace('\t', '\\t'))



get_averages_reverted()














