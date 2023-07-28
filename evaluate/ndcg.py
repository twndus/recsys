import numpy as np

# def cumulated_gain(relevance, topn):
#     return sum(relevance[:topn])

def discounted_cumulated_gain(relevance, topn):
    discounted = [np.log2(i+1) for i in range(1, topn+1)]
    return sum(relevance[:topn]/discounted)

def ndcg(answer, pred, topn):
    # answer.sort(reverse=True)
    # pred.sort(reverse=True)
    idcg = discounted_cumulated_gain(answer, topn)
    dcg = discounted_cumulated_gain(pred, topn)
    return dcg/idcg

# def average_ndcg(users, pred, gt, topn, num_user):

#     # 유저별로 결과를 모아줄 리스트 생성 (gt에는 1,0, pred에는 예측 확률)
#     gt_list = [[] for x in range(num_user)]
#     pred_list = [[] for x in range(num_user)]
#     # binary = [[] for x in range(all_user)]

#     for u,p,g in zip(users, pred, gt):
#         gt_list[u].append(g)
#         pred_list[u].append(p)

#     gt_list = np.array(gt_list)

#     print(pred_list, gt_list)

#     # 대응 정렬
#     # pred_sorted = np.sort(pred_list, reversed=True) # [2, 5, 10, 13]
#     # pred_sorted_index = np.argsort(pred_list, axis=1) # [3, 1, 0, 2]
#     # gt_sorted = [gt_list[i] for i in pred_sorted_index] # [6, 4, 2, 5]
#     # gt_sorted = np.take_along_axis(gt_list, pred_sorted_index, axis=1)

#     gt_sorted = []
#     for i, sublist in enumerate(pred_list):
#         # sorted_sublist = sorted(sublist, reverse=True)
#         pred_sorted_index = np.argsort(sublist)[::-1]
#         gt_sorted.append(np.take_along_axis(np.array(gt_list[i]), pred_sorted_index, axis=0))

#     print(gt_sorted)

#     sum_ndcg = 0

#     ideal_result = np.array([1] + [0]*(topn-1))
#     common_idcg = discounted_cumulated_gain(ideal_result, topn)
#     print(common_idcg)

#     for g in gt_sorted:
#         dcg = discounted_cumulated_gain(g, topn)
#         sum_ndcg += dcg/common_idcg

#     return sum_ndcg/gt_list.shape[0]


def average_ndcg(users, pred, gt, topn, num_user):

    # 유저별로 결과를 모아줄 리스트 생성 (gt에는 1,0, pred에는 예측 확률)
    gt_list = [[] for x in range(num_user)]
    pred_list = [[] for x in range(num_user)]

    for u,p,g in zip(users, pred, gt):
        gt_list[u].append(g)
        pred_list[u].append(p[0])

    gt_list = np.array(gt_list)
    pred_list = np.array(pred_list)

    # 대응 정렬
    pred_sorted_index = np.argsort(pred_list, axis=1)
    gt_sorted = np.take_along_axis(gt_list, pred_sorted_index, axis=1)

    sum_ndcg = 0

    ideal_result = np.array([1] + [0]*(topn-1))
    common_idcg = discounted_cumulated_gain(ideal_result, topn) # 1일껄?

    for g in gt_sorted:
        # sum_ndcg += ndcg(ideal_result, g[::-1], topn)
        dcg = discounted_cumulated_gain(g[::-1], topn) # g가 현재 오름차순으로 되어 있음
        sum_ndcg += dcg/common_idcg

    return sum_ndcg/num_user