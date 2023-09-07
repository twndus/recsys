import numpy as np

def average_hr(users, pred, gt, topn, all_users):

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

    sum_hits = 0
    hit_list = []
    for g in gt_sorted:
        hits = binary_hr(g[::-1], topn) # g가 현재 오름차순으로 되어 있음
        sum_hits += hits
        hit_list.append(hits)

    return sum_hits/len(all_users), hit_list

# def binary_hr(array, topn):
#     return array[:topn].sum()