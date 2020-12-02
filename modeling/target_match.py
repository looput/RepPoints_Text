import tensorflow as tf
from tensorflow.python.keras.backend import dtype

from tensorpack.models import Conv2D, FullyConnected, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized_method

from config import config as cfg
from utils.box_ops import pairwise_iou

from .model_box import decode_bbox_target, encode_bbox_target
from .backbone import GroupNorm


@under_name_scope()
def proposal_metrics(iou):
    """
    Add summaries for RPN proposals.

    Args:
        iou: nxm, #proposal x #gt
    """
    # find best roi for each gt, for summary only
    best_iou = tf.reduce_max(iou, axis=0)
    mean_best_iou = tf.reduce_mean(best_iou, name='best_iou_per_gt')
    summaries = [mean_best_iou]
    with tf.device('/cpu:0'):
        for th in [0.3, 0.5]:
            recall = tf.truediv(
                tf.count_nonzero(best_iou >= th),
                tf.size(best_iou, out_type=tf.int64),
                name='recall_iou{}'.format(th))
            summaries.append(recall)
    add_moving_summary(*summaries)

@under_name_scope()
def match_to_target(boxes, gt_boxes, gt_labels):
    """
    Sample some boxes from all proposals for training.
    #fg is guaranteed to be > 0, because ground truth boxes will be added as proposals.

    Args:
        boxes: nx4 region proposals, floatbox
        gt_boxes: mx4, floatbox
        gt_labels: m, int32

    Returns:
        A BoxProposals instance, with:
            sampled_boxes: tx4 floatbox, the rois
            sampled_labels: t int64 labels, in [0, #class). Positive means foreground.
            fg_inds_wrt_gt: #fg indices, each in range [0, m-1].
                It contains the matching GT of each foreground roi.
    """
    iou = pairwise_iou(boxes, gt_boxes)     # nxm
    proposal_metrics(iou)

    # add ground truth as proposals as well
    # boxes = tf.concat([boxes, gt_boxes], axis=0)    # (n+m) x 4
    # iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)   # (n+m) x m
    # #proposal=n+m from now on

    def random_sample_fg_bg(iou):
        fg_mask = tf.cond(tf.shape(iou)[1] > 0,
                          lambda: tf.reduce_max(iou, axis=1) >= cfg.FRCNN.FG_THRESH,
                          lambda: tf.zeros([tf.shape(iou)[0]], dtype=tf.bool))

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        num_fg = tf.minimum(int(
            cfg.FRCNN.BATCH_PER_IM * cfg.FRCNN.FG_RATIO),
            tf.size(fg_inds), name='num_fg')
        fg_inds = tf.random.shuffle(fg_inds)[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(
            cfg.FRCNN.BATCH_PER_IM - num_fg,
            tf.size(bg_inds), name='num_bg')
        # bg_inds = tf.random.shuffle(bg_inds)[:num_bg]

        add_moving_summary(num_fg, num_bg)
        return fg_inds, bg_inds
    
    def sample_fg_bg(iou):
        fg_mask = tf.cond(tf.shape(iou)[1] > 0,
                          lambda: tf.reduce_max(iou, axis=1) >= 0.5,
                          lambda: tf.zeros([tf.shape(iou)[0]], dtype=tf.bool))

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        # num_fg = tf.minimum(int(
        #     cfg.FRCNN.BATCH_PER_IM * cfg.FRCNN.FG_RATIO),
        #     tf.size(fg_inds), name='num_fg')
        # fg_inds = tf.random.shuffle(fg_inds)[:num_fg]
        bg_mask = tf.cond(tf.shape(iou)[1] > 0,
                          lambda: tf.reduce_max(iou, axis=1) <= 0.4,
                          lambda: tf.zeros([tf.shape(iou)[0]], dtype=tf.bool))
        bg_inds = tf.reshape(tf.where(bg_mask), [-1])
        # num_bg = tf.minimum(
        #     cfg.FRCNN.BATCH_PER_IM - num_fg,
        #     tf.size(bg_inds), name='num_bg')
        # bg_inds = tf.random.shuffle(bg_inds)[:num_bg]

        # add_moving_summary(num_fg, num_bg)
        sample_mask = tf.logical_or(fg_mask,bg_mask)
        ignore_inds = tf.reshape(tf.where(tf.logical_not(sample_mask)), [-1])
        return fg_inds, bg_inds,ignore_inds

    fg_inds, bg_inds,ignore_inds = sample_fg_bg(iou)
    # fg,bg indices w.r.t proposals

    best_iou_ind = tf.cond(tf.shape(iou)[1] > 0,
                           lambda: tf.argmax(iou, axis=1),   # #proposal, each in 0~m-1
                           lambda: tf.zeros([tf.shape(iou)[0]], dtype=tf.int64))
    fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)   # num_fg

    # TODO 增加对ignore样本的处理
    return AssignResult(
        fg_inds,
        bg_inds,
        fg_inds_wrt_gt,
        ignore_inds
    )

def assign_to_target(proposal,assign_result,gt_labels,gt_polygons):
    labels = tf.zeros(tf.shape(proposal)[0],dtype=tf.float32)
    pos_labels = tf.gather(gt_labels,assign_result.fg_gt_inds)
    labels = tf.tensor_scatter_nd_update(labels,tf.expand_dims(assign_result.fg_inds,-1),pos_labels)

    ignores = tf.zeros_like(assign_result.ignore_inds,dtype=tf.float32)
    labels = tf.tensor_scatter_nd_update(labels,tf.expand_dims(assign_result.ignore_inds,-1),ignores)

    polygons = tf.zeros_like(proposal,dtype=tf.float32)
    pos_polygons = tf.gather(gt_polygons,assign_result.fg_gt_inds)
    polygons = tf.tensor_scatter_nd_update(polygons,tf.expand_dims(assign_result.fg_inds,-1),pos_polygons)

    # TODO 添加权重
    return labels,polygons


class AssignResult(object):
    '''用来保存match 的结果，主要为index
    '''
    def __init__(self,fg_inds,bg_inds,fg_gt_inds,ignore_inds=None):
        ''' 
        Input:
            fg_inds 是指proposal中正样本的索引
            bg_inds 是指proposal中负样本的索引
            fg_gt_inds 是指正样本匹配的GT的索引
        '''
        self.fg_inds = fg_inds
        self.bg_inds = bg_inds
        self.fg_gt_inds = fg_gt_inds
        self.ignore_inds = ignore_inds


class BoxProposals(object):
    """
    A structure to manage box proposals and their relations with ground truth.
    """
    def __init__(self, boxes, labels=None, fg_inds_wrt_gt=None):
        """
        Args:
            boxes: Nx4
            labels: N, each in [0, #class), the true label for each input box
            fg_inds_wrt_gt: #fg, each in [0, M)

        The last four arguments could be None when not training.
        """
        for k, v in locals().items():
            if k != 'self' and v is not None:
                setattr(self, k, v)

    @memoized_method
    def fg_inds(self):
        """ Returns: #fg indices in [0, N-1] """
        return tf.reshape(tf.where(self.labels > 0), [-1], name='fg_inds')

    @memoized_method
    def fg_boxes(self):
        """ Returns: #fg x4"""
        return tf.gather(self.boxes, self.fg_inds(), name='fg_boxes')

    @memoized_method
    def fg_labels(self):
        """ Returns: #fg"""
        return tf.gather(self.labels, self.fg_inds(), name='fg_labels')