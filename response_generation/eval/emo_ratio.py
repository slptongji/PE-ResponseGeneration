import nltk
from nltk.translate.bleu_score import sentence_bleu
from scipy import stats
import statistics


def get_emo_words():
    emo_dictionary = {'anger':0, 'disgust':1, 'fear':2, 'joy':3, 'neutral':4, 'sadness':5, 'surprise':6}
    emo_label  = ['anger','disgust','fear','fear','joy','neutral','sadness', 'surprise']

    emo_words_total = []
    for emo in emo_label:
        emo_words = open(f"./Dictionary/{emo}.txt", "r", encoding='utf-8').readlines()[0].strip().split(',')
        emo_words_total.extend(emo_words)
    return emo_words_total

f=open('./data/persona_f1_nel2_6.txt','r',encoding='utf-8')

def get_tokens(file):
    lines = [l.strip() for l in file.readlines()]
    tokens = []
    for line in lines:
        line_l = line.split(' ')
        tokens.extend(line_l)
    return tokens

def cal_ratio(emo_words,tokens):
    cnt = 0
    for emo_word in emo_words:
        if emo_word in tokens:
            cnt += 1
    ratio = cnt / len(tokens)
    print(f"emo_ratio: {ratio}")

def avg_emo(file,emo_words):
    lines = [l.strip() for l in file.readlines()]
    print(len(lines))
    res = [0]*len(lines)
    for emo_word in emo_words:
        for i in range(len(lines)):
            line_l = lines[i].split(' ')
            if emo_word in line_l:
                res[i]+=1
    result = []
    for i in range(len(res)):
        if len(lines[i])==0:
            result.append(0.0)
        else:
            result.append(res[i]/len(lines[i]))


    return result

def cal_group(scores_modelA, scores_modelB):

    t_statistic, p_value = stats.ttest_ind(scores_modelA, scores_modelB,alternative='greater')

    df = len(scores_modelA) + len(scores_modelB) - 2
    t_critical = stats.t.ppf(0.95, df)  # 计算单侧t临界值
    mean_diff = sum(scores_modelA) / len(scores_modelA) - sum(scores_modelB) / len(scores_modelB)
    pooled_std = ((len(scores_modelA) - 1) * stats.tstd(scores_modelA) ** 2 +
                  (len(scores_modelB) - 1) * stats.tstd(scores_modelB) ** 2) / (df) ** 0.5
    margin_error = t_critical * pooled_std * (1 / len(scores_modelA) + 1 / len(scores_modelB)) ** 0.5
    confidence_interval = (mean_diff - margin_error, float('inf'))  # 单侧置信区间，上限为正无穷

    sample_std1 = statistics.stdev(scores_modelA)
    sample_std2 = statistics.stdev(scores_modelB)
    print("标准差",sum(scores_modelA)/len(scores_modelA),sum(scores_modelB)/len(scores_modelB))
    print("t统计量:", t_statistic)
    print("p值:", p_value)
    print("95%置信区间:", confidence_interval)


ff1=open('./data/persona_f1_n_400.txt','r',encoding='utf-8')
ff2=open('./data/gpt4_result.txt','r',encoding='utf-8')

emo_words = get_emo_words()
res1 = avg_emo(ff1,emo_words)
res2 = avg_emo(ff2,emo_words)
cal_group(res1,res2)
#tokens = get_tokens(ff)
#emo_words = get_emo_words()
#cal_ratio(emo_words,tokens)
