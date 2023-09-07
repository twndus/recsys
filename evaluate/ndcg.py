import numpy as np

def discounted_cumulated_gain(relevance, topn):
    discounted = [np.log2(i+1) for i in range(1, topn+1)]
    return sum(relevance[:topn]/discounted)

def ndcg(answer, pred, topn):
    idcg = discounted_cumulated_gain(answer, topn)
    dcg = discounted_cumulated_gain(pred, topn)
    return dcg/idcg

def average_ndcg(users, pred, gt, topn, all_users):
    
    # 유저 ID와 리스트 인덱스를 맵핑하기 위한 맵
    user_map = {u:i for i,u in enumerate(all_users)}

    # 유저별로 결과를 모아줄 리스트 생성 (gt에는 1,0, pred에는 예측 확률)
    gt_list = [[] for x in range(len(all_users))]
    pred_list = [[] for x in range(len(all_users))]

    for u,p,g in zip(users, pred, gt):
        gt_list[user_map[u]].append(g)
        pred_list[user_map[u]].append(p[0])

    gt_list = np.array(gt_list)
    pred_list = np.array(pred_list)

    # 대응 정렬
    pred_sorted_index = np.argsort(pred_list, axis=1)
    gt_sorted = np.take_along_axis(gt_list, pred_sorted_index, axis=1)

    sum_ndcg = 0

    ideal_result = np.array([1] + [0]*(topn-1))
    common_idcg = discounted_cumulated_gain(ideal_result, topn)

    for g in gt_sorted:
        dcg = discounted_cumulated_gain(g[::-1], topn) # g가 현재 오름차순으로 되어 있음
        sum_ndcg += dcg/common_idcg

    return sum_ndcg/len(all_users)