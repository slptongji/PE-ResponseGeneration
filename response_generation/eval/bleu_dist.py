import nltk
from nltk.translate.bleu_score import sentence_bleu
from scipy import stats
import statistics

nltk.download('punkt')
def distine_1_2(tokens):
    total_tokens = len(tokens)

    # Distinct-1
    distinct_1 = len(set(tokens)) / total_tokens

    # Distinct-2
    #distinct_2 = len(set(zip(tokens[:-1], tokens[1:]))) / (total_tokens - 1)
    bigrams = []

    for i in range(len(tokens) - 1):
        bigrams.append(tokens[i] + ' ' + tokens[i + 1])

    distink_2 = len(set(bigrams)) * 1.0 / len(bigrams)

    return distinct_1, distink_2

def get_dist(res):
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for r in res:
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len


def calculate_bleu(reference,candidate):
    reference = [reference]
    candidate = candidate
    return sentence_bleu(reference,candidate)

def eval_distinct(corpus):
    unigrams = []
    bigrams = []
    distinct_1_scores = []
    distinct_2_scores = []
    for rep in corpus:
        rep = rep.strip()
        temp = rep.split(' ')
        # 处理只有一个单词的句子情况
        if len(temp) == 1:
            distinct_1_scores.append(1.0)  # 单个单词时，Dist - 1为1
            distinct_2_scores.append(0.0)  # 无法形成二元组，Dist - 2为0
            continue
        unigrams += temp
        for i in range(len(temp) - 1):
            bigrams.append(temp[i] + ' ' + temp[i + 1])
        unigramss = set(temp)
        bigramss = set([temp[i] + ' ' + temp[i + 1] for i in range(len(temp) - 1)])
        distinct_1_scores.append(len(unigramss) * 1.0 / len(temp))
        distinct_2_scores.append(len(bigramss) * 1.0 / (len(temp) - 1))
    distink_1 = len(set(unigrams)) * 1.0 / len(unigrams)
    distink_2 = len(set(bigrams)) * 1.0 / len(bigrams)

    return distink_1, distink_2,distinct_1_scores, distinct_2_scores

def cal(true_file, test_file):

    total_distince_1 = 0
    total_distince_2 = 0
    total_bleu = 0

    tr_f = open(true_file,'r',encoding="utf-8")
    te_f = open(test_file,'r')

    tr_lines = [l.strip() for l in tr_f.readlines()]
    te_lines = [l.strip() for l in te_f.readlines()]

    ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = get_dist(te_lines)
    '''
    print(f"ma_dist1: {ma_dist1}")
    print(f"ma_dist2: {ma_dist2}")
    print(f"mi_dist1: {mi_dist1}")
    print(f"mi_dist2: {mi_dist2}")
    '''

    distink_1, distink_2,dist1_l,dist2_l = eval_distinct(te_lines)

    print(f"distink_1: {distink_1},{sum(dist1_l)/len(dist1_l)}")
    print(f"distink_2: {distink_2},{sum(dist2_l)/len(dist2_l)}")


    bleu_l = []
    for tr,te in zip(tr_lines,te_lines):

        tr_t = nltk.word_tokenize(tr)
        te_t = nltk.word_tokenize(te)
        #dist_1,dist_2 = distine_1_2(te_t)
        bleu_s = calculate_bleu(tr_t,te_t)
        bleu_l.append(bleu_s)


        #total_distince_1 += dist_1
        #total_distince_2 += dist_2
        total_bleu += bleu_s


    #avg_dist_1 = total_distince_1 / len(tr_lines)
    #avg_dist_2 = total_distince_2 / len(tr_lines)
    avg_bleu = total_bleu / len(tr_lines)
    #print(f"Average Distinct-1: {avg_dist_1}")
    #print(f"Average Distinct-2: {avg_dist_2}")
    print(f"Average BLEU Score: {avg_bleu}"+'\n')


    return bleu_l,dist1_l,dist2_l

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
    print("标准差",sample_std1,sample_std2)
    print("t统计量:", t_statistic)
    print("p值:", p_value)
    print("95%置信区间:", confidence_interval)

bleu_l_p,dist1_l_p,dist2_l_p = cal("data/answer.txt","data/persona_f1_n_400.txt")
bleu_l_g,dist1_l_g,dist2_l_g = cal("data/answer.txt","data/gpt4_result.txt")
cal_group(bleu_l_p,bleu_l_g)
