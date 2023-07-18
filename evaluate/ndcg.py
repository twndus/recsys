import numpy as np

def cumulated_gain(relevance, p):
    return sum(relevance[:p])

def discounted_cumulated_gain(relevance, p):
    discounted = [np.log2(i+1) for i in range(p)]
    return sum(relevance[:p]/discounted)/p

def ndcg(answer, pred, p):
    answer.sort(reverse=True)
    pred.sort(reverse=True)
    idcg = discounted_cumulated_gain(answer, p)
    dcg = discounted_cumulated_gain(pred, p)
    return dcg/idcg