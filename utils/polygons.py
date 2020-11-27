from collections import Sequence

import math
import matplotlib.pyplot as plt
import numpy as np


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale


def show_ann(coco, img, ann_info):
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    coco.showAnns(ann_info)
    plt.show()


# FIXME still have bug, in img719.jpg, a norm text be ignore
def spilt_point(pts):

    if pts.shape[0]==4:
        # sort the point order.
        # the clockwise vec x coord is greater to 0

        # for ICDAR15 some vechile sample, adjust the order
        # 将点都分布在长边之后，结果会发生下降，这个有bug的版本反而结果高些
        if np.linalg.norm(pts[0,:]-pts[1,:])>np.linalg.norm(pts[0,:]-pts[3,:]):
            pts = np.concatenate((pts[:2,:],pts[2:,:]),0)
        else:
            pts = np.concatenate((pts[(1,2),:], pts[(3,0),:]),0)
        
        # sort the point order.
        # the clockwise vec x coord is greater to 0
        if (pts[1]-pts[0])[0] < 0:
            pts = np.concatenate((pts[2:,:],pts[:2,:]),0)
        return pts[:2,:],pts[2:,:]
    # calculate the angle between each point

    cos_values=[]
    num=pts.shape[0]
    for idx in range(num):
        ii=idx
        vec1=(pts[(ii+1)%num,:]-pts[ii,:])
        vec2=(pts[(ii+num-1)%num]-pts[ii,:])
        cos_value=(vec1*vec2).sum()/(np.linalg.norm(vec1)*np.linalg.norm(vec2)+1e-6)
        cos_value_0=np.abs(cos_value)
        
        ii=(idx+1)%num
        vec1=(pts[(ii+1)%num,:]-pts[ii,:])
        vec2=(pts[(ii+num-1)%num]-pts[ii,:])
        cos_value=(vec1*vec2).sum()/(np.linalg.norm(vec1)*np.linalg.norm(vec2)+1e-6)
        cos_value=np.abs(cos_value)

        cos_value=(cos_value+cos_value_0)/2
        cos_values.append(cos_value)
    # find the 3 min value
    cos_val=np.array(cos_values)

    index_chooses=[(0,1),(0,2),(0,3),(1,2),(1,3)]
    for idx,idx_choos in enumerate(index_chooses):
        min_index=np.argsort(cos_val)[idx_choos,]
        min_index_clock=np.sort(min_index)
        if not (abs(min_index_clock[0] - min_index_clock[1]) == 1 or abs((min_index_clock[0] - min_index_clock[1] + num) % num) == 1):
            break
        if idx==len(index_chooses)-1:
            return None,None

    min_index_clock=np.array((min_index_clock[0],(min_index_clock[0]+1)%num,min_index_clock[1],(min_index_clock[1]+1)%num),dtype=np.int32)

    min_index_final=min_index_clock[(1,2,3,0),]
    # corner_point=pts[min_index_clock,:]

    # if np.linalg.norm(corner_point[0]-corner_point[1])>np.linalg.norm(corner_point[2]-corner_point[3]):
    #     min_index_final=min_index_clock
    # else:
    #     min_index_final=min_index_clock[(1,2,3,0),]

    split_0,split_1=[],[]
    iidx,iidx_e=min_index_final[0],min_index_final[1]
    while iidx!=iidx_e:
        split_0.append(pts[iidx,:])
        iidx=(iidx+1)%num
    split_0.append(pts[iidx_e,:])
    
    iidx,iidx_e=min_index_final[2],min_index_final[3]
    while iidx!=iidx_e:
        split_1.append(pts[iidx,:])
        iidx=(iidx+1)%num
    split_1.append(pts[iidx_e,:])
    
    # the clockwise vec x coord is greater to 0
    if (split_0[-1]-split_0[0])[0]>0:
        left_up=split_0
        right_down=split_1
    else:
        left_up=split_1
        right_down=split_0
        
    return np.array(left_up),np.array(right_down)

def expand_point(pts,num_exp=7):
    sh=pts.shape
    if sh[0]<4:
        pts_tile=np.tile(pts[-1:,:],[num_exp*2-sh[0],1])
        pts=np.concatenate((pts,pts_tile),axis=0)
        return pts

    up_point,down_point=spilt_point(pts)
    if up_point is None:
        if sh[0]>num_exp*2:
            return pts[:num_exp*2,:]
        else:
            pts_tile=np.tile(pts[-1:,:],[num_exp*2-sh[0],1])
            pts=np.concatenate((pts,pts_tile),axis=0)
            return pts

    def expand(point_base):
        #  all point must in clock or unclockwise
        dis=np.linalg.norm(point_base[1:,:]-point_base[:-1,:],axis=1)
        up_len=dis.sum()
        dis_per=max(up_len/(num_exp-1),1)

        line_seg=[0]
        for i in range(dis.shape[0]):
            line_seg.append(line_seg[i]+dis[i])
        
        line_seg=np.array(line_seg)

        exp_list=[]
        for i in range(num_exp):
            cur_pos=dis_per*i
            
            dis=line_seg-cur_pos
            index=np.argsort(np.abs(line_seg-cur_pos))
            if dis[index[0]]*dis[index[1]]<0:
                a_idx,b_idx=index[0],index[1]
            elif len(dis)>2:
                a_idx,b_idx=index[0],index[2]
            else:
                a_idx,b_idx=index[0],index[1]

            point_exp=(point_base[a_idx,:]-point_base[b_idx,:])*(cur_pos-line_seg[b_idx])/(line_seg[a_idx]-line_seg[b_idx]+1e-6)+point_base[b_idx,:]

            exp_list.append(point_exp)
        
        return exp_list
    
    up_expand=expand(up_point)
    down_expand=expand(down_point)
    
    point_expand=np.array(up_expand+down_expand)
    return point_expand

def GetPts_td(line):
    pts_lst = line.split()
    hard = int(pts_lst[1])
    x = int(pts_lst[2])
    y = int(pts_lst[3])
    w = int(pts_lst[4])
    h = int(pts_lst[5])
    theta = float(pts_lst[6])
    x1 = math.cos(theta) * (-0.5 * w) - math.sin(theta) * (-0.5 * h) + x + 0.5 * w
    y1 = math.sin(theta) * (-0.5 * w) + math.cos(theta) * (-0.5 * h) + y + 0.5 * h
    x2 = math.cos(theta) * (0.5 * w) - math.sin(theta) * (-0.5 * h) + x + 0.5 * w
    y2 = math.sin(theta) * (0.5 * w) + math.cos(theta) * (-0.5 * h) + y + 0.5 * h
    x3 = math.cos(theta) * (0.5 * w) - math.sin(theta) * (0.5 * h) + x + 0.5 * w
    y3 = math.sin(theta) * (0.5 * w) + math.cos(theta) * (0.5 * h) + y + 0.5 * h
    x4 = math.cos(theta) * (-0.5 * w) - math.sin(theta) * (0.5 * h) + x + 0.5 * w
    y4 = math.sin(theta) * (-0.5 * w) + math.cos(theta) * (0.5 * h) + y + 0.5 * h
    pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
    return pts, hard

# import pyclipper
# import Polygon as plg

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        
        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 3:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bboxes.append(shrinked_bbox)
    
    return np.array(shrinked_bboxes)

def expand(point_base,expand,seg_loc):
        #  all point must in clock or unclockwise
        dis=np.linalg.norm(point_base[1:,:]-point_base[:-1,:],axis=1)
        up_len=dis.sum()

        isEnd=False
        if seg_loc[0]>=up_len:
            return True,None
        elif seg_loc[1]>up_len:
            seg_loc[1]=up_len
            isEnd = True
        
        seg_len = seg_loc[1]-seg_loc[0]

        dis_per=max(seg_len/(expand-1),1)

        line_seg=[0]
        for i in range(dis.shape[0]):
            line_seg.append(line_seg[i]+dis[i])
        line_seg=np.array(line_seg)

        exp_list=[]
        for i in range(expand):
            cur_pos = seg_loc[0] + dis_per*i
            
            dis=line_seg-cur_pos
            index=np.argsort(np.abs(line_seg-cur_pos))
            if dis[index[0]]*dis[index[1]]<0:
                a_idx,b_idx=index[0],index[1]
            elif len(dis)>2:
                a_idx,b_idx=index[0],index[2]
            else:
                a_idx,b_idx=index[0],index[1]

            point_exp=(point_base[a_idx,:]-point_base[b_idx,:])*(cur_pos-line_seg[b_idx])/(line_seg[a_idx]-line_seg[b_idx]+1e-6)+point_base[b_idx,:]

            exp_list.append(point_exp)
        
        return isEnd,exp_list