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
    angles=[]
    for line in lines:
        if line == '':
            continue
        line = line.split(';')
        bbox=line[:-1]
        angle=line[-1]
        if len(bbox) % 2 == 1:
            print(path)
        bbox = [(int)(x) for x in bbox]
        bboxes.append(bbox)
        angles.append(angle)
    return bboxes,angles

def get_gt(path,use_angle=False):
    reader = open(path).readlines()
    bboxes = []
    angles= []
    for line in reader:
        line=line.replace("\ufeff", "")
        if use_angle:
            parts = line.strip().split(';',10)
            text=parts[-2]
            polygon = parts[1:9]
            polygon = [int(p) for p in polygon]
            angle = parts[-1].split('@')[-1]
        else:
            # text=parts[-1]
            parts = line.strip().split(';')
            num_parts = parts[1:-1]
            num_parts = [nn for nn in num_parts if nn.isdigit()]
            num_parts = num_parts[:(len(num_parts)//2)*2]
            text = ''.join(parts[1+len(num_parts):])
            polygon = [int(p) for p in num_parts]
            angle = 0.
        bboxes.append(polygon)
        angles.append(angle)

    return bboxes,angles

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
        preds,angles = get_pred(pred_path)
        gt_path = gt_root+'/' + pred_path.split('/')[-1]
        gts,gt_angles = get_gt(gt_path)
        npos += len(gts)
        
        cover = set()
        for pred_id, (pred,angle) in enumerate(zip(preds,angles)):
            pred = np.array(pred)
            pred = pred.reshape(int(pred.shape[0] / 2), 2)
            # if pred.shape[0] <= 2:
            #     continue

            pred_p = plg.Polygon(pred)
            
            flag = False
            flag_rec = False
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

                        flag_rec=abs(int(gt_angles[gt_id])-int(angle))<45
                        # print(int(gt_angles[gt_id]),int(angle))

            if flag:
                tp += 1.0
            else:
                fp += 1.0

            if flag_rec:
                tp_rec+=1.0
            else:
                fp_rec+=1.0

    print(tp, fp, npos)
    precision = tp / (tp + fp)
    recall = tp / npos
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    
    precision_rec = tp_rec / (tp_rec + fp_rec)
    recall_rec = tp_rec / npos
    hmean_rec = 0 if (precision_rec + recall_rec) == 0 else 2.0 * precision_rec * recall_rec / (precision_rec + recall_rec)

    print('p: %.4f, r: %.4f, f: %.4f'%(precision, recall, hmean))
    # print('p: %.4f, r: %.4f, f: %.4f, p: %.4f, r: %.4f, f_angle: %.4f'%(precision, recall, hmean,precision_rec,recall_rec,hmean_rec))
    
    return {'precision':precision,'recall':recall,'hmean':hmean}