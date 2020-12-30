from matplotlib.pyplot import text
from .file_util import read_file,read_dir
import Polygon as plg
import numpy as np
import re
import string

# pred_root = '../../outputs/submit_ctw1500/'
# gt_root = '../../data/CTW1500/test/text_label_curve/'

def transcription_match(transGt,transDet,specialCharacters=str('!?.:,*"()·[]/\''),onlyRemoveFirstLastCharacterGT=False):

    if onlyRemoveFirstLastCharacterGT:
        #special characters in GT are allowed only at initial or final position
        if (transGt==transDet):
            return True        

        if specialCharacters.find(transGt[0])>-1:
            if transGt[1:]==transDet:
                return True

        if specialCharacters.find(transGt[-1])>-1:
            if transGt[0:len(transGt)-1]==transDet:
                return True

        if specialCharacters.find(transGt[0])>-1 and specialCharacters.find(transGt[-1])>-1:
            if transGt[1:len(transGt)-1]==transDet:
                return True
        return False
    else:
        transGt = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), transGt.lower()))
        transDet = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), transDet.lower()))
        return transGt == transDet

def get_pred(path):
    lines = read_file(path).split('\n')
    bboxes = []
    texts=[]
    for line in lines:
        if line == '':
            continue
        line = line.split(';')
        bbox=line[:-1]
        text=line[-1]
        if len(bbox) % 2 == 1:
            print(path)
        bbox = [(int)(x) for x in bbox]
        bboxes.append(bbox)
        texts.append(text)
    return bboxes,texts

def get_gt(path):
    reader = open(path).readlines()
    bboxes = []
    texts = []
    ignores = []
    for line in reader:
        line=line.replace("\ufeff", "")
        # text=parts[-1]
        parts = line.strip().split(';')
        num_parts = parts[1:-1]
        num_parts = [nn for nn in num_parts if nn.isdigit()]
        num_parts = num_parts[:(len(num_parts)//2)*2]
        text = ''.join(parts[1+len(num_parts):])
        # print(text)
        polygon = [int(p) for p in num_parts]
        bboxes.append(polygon)
        texts.append(text)
        ignores.append(1 if ('dots' in text or 'arc_seal' in text or 'rect_seal' in text) else 0)

    return bboxes,texts,ignores

def detection_filtering(preds,pred_texts,gts,gt_texts,ignores,threshold=0.5):
    for gt_id, gt in enumerate(gts):
        if ignores[gt_id]==1:
            gt = np.array(gt)
            gt = gt.reshape(int(gt.shape[0] / 2), 2)
            gt_p = plg.Polygon(gt)

            for det_id, pred in enumerate(preds):
                pred = np.array(pred)
                pred = pred.reshape(int(pred.shape[0] / 2), 2)
                pred_p = plg.Polygon(pred)
                iou = get_intersection(pred_p, gt_p)/get_union(pred_p,gt_p)

                if iou > threshold:
                    preds[det_id]=None
                    pred_texts[det_id]=None
                    # TODO 一旦有样本匹配上，是不是该GT就不应参与匹配
                    break

            preds = [item for item in preds if item != None]
            pred_texts = [item for item in pred_texts if item != None]
            gts[gt_id]=None
            gt_texts[gt_id]=None
    gts = [item for item in gts if item!=None] 
    gt_texts = [item for item in gt_texts if item!=None] 
    return preds,pred_texts,gts,gt_texts

def get_union(pD,pG):
    areaA = pD.area();
    areaB = pG.area();
    return areaA + areaB - get_intersection(pD, pG);        

def get_intersection(pD,pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

def eval_result(pred_root,gt_root):
    th = 0.5
    pred_list = read_dir(pred_root)

    tp, fp, npos = 0, 0, 0
    tp_rec, fp_rec, npos = 0, 0, 0
    
    for pred_path in pred_list:
        preds,texts = get_pred(pred_path)
        gt_path = gt_root+'/' + pred_path.split('/')[-1]
        gts,gt_texts,ignores = get_gt(gt_path)
        preds,texts,gts,gt_texts = detection_filtering(preds,texts,gts,gt_texts,ignores)
        npos += len(gts)
        
        cover = set()
        for pred_id, (pred,text) in enumerate(zip(preds,texts)):
            pred = np.array(pred)
            pred = pred.reshape(int(pred.shape[0] / 2), 2)
            # if pred.shape[0] <= 2:
            #     continue

            pred_p = plg.Polygon(pred)
            
            flag = False
            # flag_rec = False
            for gt_id, gt in enumerate(gts):
                gt = np.array(gt)
                gt = gt.reshape(int(gt.shape[0] / 2), 2)
                gt_p = plg.Polygon(gt)

                union = get_union(pred_p, gt_p)
                inter = get_intersection(pred_p, gt_p)

                if inter * 1.0 / union >= th:
                    if gt_id not in cover:
                        flag = True
                        cover.add(gt_id)

                        # flag_rec=abs(int(gt_angles[gt_id])-int(angle))<45
                        # print(int(gt_angles[gt_id]),int(angle))
            if flag:
                tp += 1.0
            else:
                fp += 1.0

            # if flag_rec:
            #     tp_rec+=1.0
            # else:
            #     fp_rec+=1.0

    print(tp, fp, npos)
    precision = tp / (tp + fp)
    recall = tp / npos
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    
    # precision_rec = tp_rec / (tp_rec + fp_rec)
    # recall_rec = tp_rec / npos
    # hmean_rec = 0 if (precision_rec + recall_rec) == 0 else 2.0 * precision_rec * recall_rec / (precision_rec + recall_rec)
    print('P, R, F, %.4f %.4f %.4f'%(precision, recall, hmean))
    # print('p: %.4f, r: %.4f, f: %.4f, p: %.4f, r: %.4f, f_angle: %.4f'%(precision, recall, hmean,precision_rec,recall_rec,hmean_rec))
    
    return {'tp':tp, 'fp':fp, 'npos':npos,'precision':precision,'recall':recall,'hmean':hmean}