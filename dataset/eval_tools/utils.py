import os
import codecs
import numpy as np

def is_contain(rect1, rect2):
    """
    如果rect2的框都在rect1 内，返回true
    """
    if (rect1[0] <= rect2[0] and rect1[1] <= rect2[1] and rect1[2] >= rect2[2] and rect1[3] >= rect2[3]):
        return True
    return False


def compute_parallel_rate(rect1, rect2):
    """
    返回两个方框在竖直方向上的平行率
    """
    # 高度相差过大的不合并
    h1, h2 = rect1[3] - rect1[1], rect2[3] - rect2[1]
    if h1 > 1.6 * h2 or h2 > 1.6 * h1:
        return 0

    if is_contain(rect1, rect2) or is_contain(rect2, rect1):
        return 0

    top = max(rect1[1], rect2[1])
    button = min(rect1[3], rect2[3])
    if button <= top:
        return 0
    return ((button - top) * 2) / float(rect1[3] + rect2[3] - rect1[1] - rect2[1])


def merge_parallel_box(rect1, rect2, threshold, rate=0.7):
    """
    如果rec1，rec2是平行的且相距在阈值内，则合并成一个框
    """
    ymin = min(rect1[0], rect2[0])
    ymax = max(rect1[2], rect2[2])
    xmin = min(rect1[1], rect2[1])
    xmax = max(rect1[3], rect2[3])
    # rect1_rate = float(rect1[2]-rect1[0])/(rect1[3]-rect1[1]+1e-6)
    # rect2_rate = float(rect2[2]-rect2[0])/(rect2[3]-rect2[1]+1e-6)
    rect1_rate,rect2_rate = 1.01,1.01
    # h1, h2 = rect1[3] - rect1[1], rect2[3] - rect2[1]
    # threshold = threshold * (h1 + h2) / 2.0
    # print(rect2[0] - rect1[2])
    if rect1[0] > rect2[0] and (rect1[0] - rect2[2]) > threshold:
        return []
    elif rect2[0] > rect1[0] and (rect2[0] - rect1[2]) > threshold:
        return []
    if compute_parallel_rate(rect1, rect2) > rate and (rect1_rate-1<0.1 or rect2_rate-1<0.1):
        return [ymin, xmin, ymax, xmax]
    if (rect1[3] <= rect2[3] and rect2[1] - 4 <= rect1[1]
            and rect2[3] + 4 >= rect1[3]) and (rect1_rate-1<0.1 or rect2_rate-1<0.1):
        return [ymin, xmin, ymax, xmax]
    if (rect2[3] <= rect1[3] and rect1[1] - 4 <= rect2[1]
            and rect1[3] + 4 >= rect2[3]) and (rect1_rate-1<0.1 or rect2_rate-1<0.1):
        return [ymin, xmin, ymax, xmax]
    return []


def merge_parallel_rect(rects, label_names, threshold, rate=0.7):
    """
    检测wordRect，如果是平行的且相距在阈值内，则合并成一个框
    """
    # print(rects, label_names)
    if rects == []:
        return []
    rects_old = rects
    rects_new = []
    label_names_new = []
    rects_new.append(rects_old[0])
    label_names_new.append(label_names[0])
    idxi = 1
    while idxi < len(rects_old):  # 取旧的框
        idxj = 0
        idxjmax = len(rects_new)
        merge_flag = False
        while idxj < idxjmax:  # 旧的框跟新的框比较
            # print(idxi, rects[idxi])
            # print(idxi, label_names[idxi])
            # import pdb
            # pdb.set_trace()
            # print('label_names[idxi]', idxi, len(label_names), len(rects), label_names[idxi], rects[idxi])
            rect = []
            # if label_names[idxi] == "del_item" or label_names_new[idxj] == "del_item":
            rect = merge_parallel_box(rects_old[idxi], rects_new[idxj], threshold, rate)
            if rect != []:
                rects_new[idxj][0:4] = rect
                label_names_new[idxj] = "text"
                merge_flag = True
                break
            idxj += 1
        # 不能合并,将该框加入新框中
        if merge_flag is False:
            rects_new.append(rects_old[idxi])
            label_names_new.append(label_names[idxi])
        idxi += 1
    # 递归直到不能继续合并为止
    
    # print(111111)
    # print(len(rects_new), len(rects_old))
    
    if len(rects_new) != len(rects_old):
        rects_new, label_names_new = merge_parallel_rect(rects_new, label_names_new, threshold, rate)
    return rects_new, label_names_new

def merge_label(res_rects):
    """
    合并
    :return:
    """

    new_lines = []
    rects = [rect[1:3]+rect[5:7] for rect in res_rects]
    label_names = [rect[9] for rect in res_rects]
    # print(rects, label_names)
    count_array = np.array(rects)
    line_height = np.mean(count_array[:, 3] - count_array[:, 1])
    
    rects, label_names_new = merge_parallel_rect(rects, label_names, threshold=Params.HROZIE_GAP_RATE * line_height, rate=Params.VETICAL_GAP_RATE)
    # print('rects=======', rects)
    for idx, rect in enumerate(rects):
        rect.insert(0, res_rects[0][0])
        rect.insert(3, rect[3])
        rect.insert(4, rect[2])
        rect.insert(7, rect[1])
        rect.insert(8, rect[6])
        rect.append("text@\n")
    # print('rects=======', rects)
    return rects

def merge(bboxes,texts,ignores,HROZIE_GAP_RATE=0.8,VETICAL_GAP_RATE=0.5):
    bboxes_ignores = [bb for idx,bb in enumerate(bboxes) if ignores[idx]]
    texts_ignores = [tt for idx,tt in enumerate(texts) if ignores[idx]]

    bboxes = [bb[:2]+bb[4:6] for idx,bb in enumerate(bboxes) if ignores[idx]==0]
    texts = [tt for idx,tt in enumerate(texts) if ignores[idx]==0]

    new_lines = []
    # print(rects, label_names)
    count_array = np.array(bboxes)
    line_height = np.mean(count_array[:, 3] - count_array[:, 1])

    # print(HROZIE_GAP_RATE * line_height)
    rects, label_names_new = merge_parallel_rect(bboxes, texts, threshold=HROZIE_GAP_RATE * line_height, rate=VETICAL_GAP_RATE)
    for idx, rect in enumerate(rects):
        rect.insert(2, rect[2])
        rect.insert(3, rect[1])
        rect.insert(6, rect[0])
        rect.insert(7, rect[5])
    # print(len(bboxes),len(rects))
    rects = rects+bboxes_ignores
    labels = label_names_new+texts_ignores
    ignores = [0 if i<len(texts) else 1 for i in range(len(rects))]

    return rects,labels,ignores
