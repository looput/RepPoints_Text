# -*- coding: utf-8 -*-
# Reimplement of Reppoint with tensorflow.
# Copyright (c) 2020, HUST loop.

import sys

# from tensorpack import tfv1 as tf
import numpy as np
import tensorflow as tf
from tensorpack import ModelDesc
from tensorpack.models import (Conv2D, Conv2DTranspose, GlobalAvgPooling,
                               l2_regularizer, layer_register, regularize_cost)
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.summary import add_moving_summary

sys.path.append('/home/lupu/project/RepPoints')

from config import config as cfg
from data import get_all_anchors, get_all_anchors_fpn
from layers.dcn.deformable_conv import DeformableConv2D
from layers.deformable_conv_layer import DeformableConv_TF
from utils.box_ops import area as tf_area

from modeling.backbone import (GroupNorm, image_preprocess, resnet_c4_backbone,
                               resnet_conv5, resnet_fpn_backbone)
from modeling.model_fpn import (fpn_model, generate_fpn_proposals,
                                multilevel_roi_align, multilevel_rpn_losses)
from utils.point_generator import PointGenerator


def ConvModule(x,out_channels,
            kernel = (3,3),
            stride = (1,1),
            with_norm=True,
            act='relu',
            name='conv'):
    with argscope(Conv2D, data_format='channels_first',
                  kernel_initializer=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_out',
                      distribution='untruncated_normal')):
        x = Conv2D(name,x,out_channels, kernel)
        if with_norm:
            x = GroupNorm(f'gn_{name}',x)
        if act!='':
            x = tf.keras.layers.Activation(act)(x)
        return x

def DefConvModule_TF(x,offset,
            out_channels,
            kernel = (3,3),
            stride = (1,1),
            with_norm=True,
            act='relu',
            name='dcn'):
    
    # this op is only support channel_last only
    x = tf.transpose(x,[0,2,3,1])
    offset = tf.transpose(offset,[0,2,3,1])
    def_conv = DeformableConv_TF(out_channels,kernel,stride,num_deformable_group=1,padding='same')
    # def_conv = DeformableConvLayer(out_channels,kernel,stride,padding='same')
    x =  def_conv(x,offset)

    if with_norm:
            x = GroupNorm(f'gn_{name}',x)
    if act!='':
        x = tf.keras.layers.Activation(act)(x)
    
    x = tf.transpose(x,[0,3,1,2])
    return x

def DefConvModule(x,offset,
            out_channels,
            kernel = (3,3),
            stride = (1,1),
            with_norm=True,
            act='relu',
            name='dcn'):
    
    # this op is only support channel_last only
    def_conv = DeformableConv2D(out_channels,
                                kernel,
                                use_bias=False,
                                padding='SAME')
    # def_conv = DeformableConvLayer(out_channels,kernel,stride,padding='same')
    x =  def_conv(x,offset)
    if with_norm:
            x = GroupNorm(f'gn_{name}',x)
    if act!='':
        x = tf.keras.layers.Activation(act)(x)
    return x


class SigleStageDet(ModelDesc):
    def preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0., trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)

        # The learning rate in the config is set for 8 GPUs, and we use trainers with average=False.
        lr = lr / 8.
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        if cfg.TRAIN.NUM_GPUS < 8:
            opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
        return opt

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        `build_graph` must create tensors of these names when called under inference context.

        Returns:
            [str]: input names
            [str]: output names
        """
        out = ['output/polygons', 'output/scores', 'output/labels']
        if cfg.MODE_MASK:
            out.append('output/masks')
        if cfg.MODE_POLYGON:
            out.append('output/polygons')
        return ['image'], out

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))
        # if "gt_masks_packed" in inputs:
        #     gt_masks = tf.cast(unpackbits_masks(inputs.pop("gt_masks_packed")), tf.uint8, name="gt_masks")
        #     inputs["gt_masks"] = gt_masks

        image = self.preprocess(inputs['image'])     # 1CHW

        features = self.backbone(image)[1:4]
        box_outs = self.box_head(image,features)

        # if self.training:
        if False:
            targets = dict(zip([inputs[k] for k in ['gt_boxes', 'gt_labels', 'gt_masks'] if k in inputs]))
            head_losses = self.head_losses(features,box_outs,targets)
            
            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
            total_cost = tf.add_n(
                 head_losses + [wd_cost], 'total_cost')
            add_moving_summary(total_cost, wd_cost)
            return total_cost
        else:
            result = self.postprocess(box_outs)
            return result

            # Check that the model defines the tensors it declares for inference
            # For existing models, they are defined in "fastrcnn_predictions(name_scope='output')"
            G = tf.get_default_graph()
            ns = G.get_name_scope()
            for name in self.get_inference_tensor_names()[1]:
                try:
                    name = '/'.join([ns, name]) if ns else name
                    G.get_tensor_by_name(name + ':0')
                except KeyError:
                    raise KeyError("Your model does not define the tensor '{}' in inference context.".format(name))


class RepPointsC4Det(ModelDesc):
    def inputs(self):
        ret = [
            tf.TensorSpec((None, None, 3), tf.float32, 'image'),
            tf.TensorSpec((None, None, cfg.RPN.NUM_ANCHOR), tf.int32, 'anchor_labels'),
            tf.TensorSpec((None, None, cfg.RPN.NUM_ANCHOR, 4), tf.float32, 'anchor_boxes'),
            tf.TensorSpec((None, 4), tf.float32, 'gt_boxes'),
            tf.TensorSpec((None,), tf.int64, 'gt_labels')]  # all > 0
        if cfg.MODE_MASK:
            ret.append(
                tf.TensorSpec((None, None, None), tf.uint8, 'gt_masks_packed')
            )   # NR_GT x height x ceil(width/8), packed groundtruth masks
        return ret

    def backbone(self, image):
        return [resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS[:3])]

    def box_head(self, image, features):
        # Multi-Level RPN Proposals]
        # strides = cfg.BOX_HEAD.STRIDES
        strides = (8,16,32)
        reppoint_outputs = [reppoints_head(pi,stride) for pi,stride in zip(features,strides)]

        mlvl_cls_out = [k[0] for k in reppoint_outputs]
        mlvl_pts_init_out = [k[1] for k in reppoint_outputs]
        mlvl_pts_refine_out = [k[2] for k in reppoint_outputs]

        return dict(
            cls_outs=mlvl_cls_out,
            pts_init_outs=mlvl_pts_init_out,
            pts_refine_outs=mlvl_pts_refine_out
        )


def _init_dcn_offset(num_points):
    dcn_kernel = int(np.sqrt(num_points))
    dcn_pad = int((dcn_kernel - 1) / 2)
    assert dcn_kernel * dcn_kernel == num_points, "The points number should be a square number."
    assert dcn_kernel % 2 == 1, "The points number should be an odd square number."
    dcn_base = np.arange(-dcn_pad, dcn_pad + 1).astype(np.float)
    dcn_base_y = np.repeat(dcn_base, dcn_kernel)
    dcn_base_x = np.tile(dcn_base, dcn_kernel)
    dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
    dcn_base_offset = tf.constant(dcn_base_offset,shape=[1,dcn_base_offset.shape[0],1,1],dtype=tf.float32)

    return dcn_base_offset

def reppoints_head(feature_map,stride,num_points=9):
    """RepPoint 分类和回归分支
        Note: 使用 channel first
        feature_map: backbone features [batch, depth ,height, width]
        stride: 使用的特征的步长 int

        Returns:
            cls_probs: [batch,nCls, H, W]  
            pts_init: [batch,(y1, x1, y2, x2, ...), H, W] 初次回归的目标框的偏移量
            pts_refine: [batch,(y1, x1, y2, x2, ...), H, W] 二次回归的目标框的偏移量
    """
    with tf.variable_scope("reppoints_head", reuse=tf.AUTO_REUSE) as scope:
        dcn_base_offset = _init_dcn_offset(9)

        cls_feat = feature_map
        pts_feat = feature_map

        for i in range(1):
            cls_feat = ConvModule(cls_feat,256,name=f'cls_conv_{i}')

        for i in range(1):
            pts_feat = ConvModule(pts_feat,256,name=f'pts_conv_{i}')

        x = ConvModule(pts_feat,256,with_norm=False,act='relu',name='pts_init_conv')
        pts_out_init = ConvModule(x,18,with_norm=False,act='',name='pts_init_out')# [b,2n,h,w]

        gradient_mul = 0.1
        pts_out_init_detach = tf.stop_gradient(pts_out_init)*(1-gradient_mul)+pts_out_init
        # classify

        dcn_offset = pts_out_init_detach - dcn_base_offset
        cls_feat_dcn = DefConvModule(cls_feat,dcn_offset,128,with_norm=False,act='relu')
        cls_out = ConvModule(cls_feat_dcn,cfg.DATA.NUM_CATEGORY,(1,1),with_norm=False,name='cls_out')

        pts_feat_dcn = DefConvModule(pts_feat,dcn_offset,128,with_norm=False,act='relu')
        pts_out_refine = ConvModule(pts_feat_dcn,2*9,(1,1),with_norm=False,name='pts_refine')

        pts_out_refine = tf.math.add(tf.stop_gradient(pts_out_init),pts_out_refine,name='pts_refine_out')

        return cls_out,pts_out_init,pts_out_refine  

    # Dense RepPoint 
    # # point refine
    # x = ConvModule(pts_feat,256,with_norm=False,act='relu',name='pts_refine_conv')(pts_feat)
    # pts_refine_field = ConvModule(x,18,with_norm=False,act='',name='pts_refine_field')(x) # [b,h,w,n*2]
    # shape = tf.shape(pts_refine_field)
    # pts_refine_field = K.reshape(pts_refine_field,(shape[0],shape[1],shape[2],-1,2))    # [b,h,w,n,2]
    # pts_out_init_detach = K.reshape(pts_out_init_detach,(shape[0],shape[1],shape[2],-1,2)) # [b,h,w,n,2]

def multiclass_nms(pts, scores):
    """
    对结果执行nms
    Args:
        pts: (n,2*np) floatbox in float32
        scores: (n,nCls)
    Returns:
        boxes: (n,5) （最后一维包含得分）
        pts: (n,np*2)
        labels: (n,)
    """
    # assert pts.shape[0] == scores.shape[0]

    # 将 pts 转换成 bbox
    pts = tf.reshape(pts,[tf.shape(pts)[0],-1,2]) # (n,np,2)
    bbox_left = tf.reduce_min(pts[:,:,0],axis=1,keepdims=True)
    bbox_right = tf.reduce_max(pts[:,:,0],axis=1,keepdims=True)
    bbox_up = tf.reduce_min(pts[:,:,1],axis=1,keepdims=True)
    bbox_bottom = tf.reduce_max(pts[:,:,1],axis=1,keepdims=True)

    boxes = tf.concat([bbox_left, bbox_up, bbox_right, bbox_bottom],axis=1) 

    inds = tf.argmax(scores,axis=1) # (n)

    # tf 对去索引支持的相当不完善...
    # https://stackoverflow.com/questions/45836241/tensorflow-tf-argmax-and-slicing
    ins_n = tf.cast(tf.shape(inds)[0],dtype=inds.dtype)
    idx = tf.stack([tf.range(ins_n), inds], axis=-1)
    scores = tf.gather_nd(scores,idx)
    labels = tf.add(inds,1)

    masks = tf.greater(scores,cfg.TEST.RESULT_SCORE_THRESH)

    filtered_scores = tf.boolean_mask(scores,masks,axis=0)
    filtered_boxes = tf.boolean_mask(boxes,masks,axis=0)
    filtered_pts = tf.boolean_mask(pts,masks,axis=0)
    filtered_labels = tf.boolean_mask(labels,masks,axis=0)

    max_coord = tf.reduce_max(filtered_boxes)
    offsets = tf.cast(filtered_labels, tf.float32) * (max_coord + 1)  # F,1
    offsets = tf.expand_dims(offsets,axis=1)
    nms_boxes = filtered_boxes + offsets
    selection = tf.image.non_max_suppression(
        nms_boxes,
        filtered_scores,
        cfg.TEST.RESULTS_PER_IM,
        cfg.TEST.FRCNN_NMS_THRESH)

    final_scores = tf.gather(filtered_scores, selection, name='scores')
    final_labels = tf.gather(labels, selection, name='labes')
    final_boxes = tf.gather(filtered_boxes, selection)
    final_pts = tf.gather(filtered_pts, selection, name='pts')

    final_scores = tf.expand_dims(final_scores,1)
    final_boxes = tf.concat((final_boxes,final_scores),1,name='boxes')
    return final_boxes,final_pts,final_labels

class RepPointsFPNDet(SigleStageDet):

    def inputs(self):
        ret = [
            tf.TensorSpec((None, None, 3), tf.float32, 'image')]
        ret.extend([
            tf.TensorSpec((None, 4), tf.float32, 'gt_boxes'),
            tf.TensorSpec((None,), tf.int64, 'gt_labels')])  # all > 0
        if cfg.MODE_MASK:
            ret.append(
                tf.TensorSpec((None, None, None), tf.uint8, 'gt_masks_packed')
            )
        if cfg.MODE_POLYGON:
            ret.append(
                tf.TensorSpec((None, None, None), tf.float32, 'gt_polygons')
            )
        return ret

    def backbone(self, image):
        c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS)
        p23456 = fpn_model('fpn', c2345)
        return p23456

    def box_head(self, image, features):
        # Multi-Level RPN Proposals]
        # strides = cfg.BOX_HEAD.STRIDES
        strides = (8,16,32)
        reppoint_outputs = [reppoints_head(pi,stride) for pi,stride in zip(features,strides)]

        mlvl_cls_out = [k[0] for k in reppoint_outputs]
        mlvl_pts_init_out = [k[1] for k in reppoint_outputs]
        mlvl_pts_refine_out = [k[2] for k in reppoint_outputs]

        return dict(
            cls_outs=mlvl_cls_out,
            pts_init_outs=mlvl_pts_init_out,
            pts_refine_outs=mlvl_pts_refine_out
        )

    def head_losses(self,box_outs,targets):
        '''
        Args:
            box_out: dict(
                cls_outs: (n,nCls,h,w)
                pts_init_outs: (n,nP,h,w) [y first]
                pts_refine_outs: (n,nP,h,w) [y first]
            )
            targets: dict(
                
            )
        returns:
            loss: a tensor
        '''
        
        pass

    def points_formate(self,pts,y_first=True):
        shape = tf.shape(pts)
        pts_reshape =  tf.reshape(pts,(shape[0],-1,2,shape[2],shape[3]))
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,:,:]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,:,:]

        pts = tf.stack((pts_x,pts_y),axis=2)
        pts = tf.reshape(pts,[tf.shape(pts)[0],-1,tf.shape(pts)[3],tf.shape(pts)[4]])
        return pts

    def postprocess(self,bbox_head_outs):
        '''
            为了和DCN适配，坐标使用y,x 后处理过程需要进行颠倒
            bbox_head_outs: dict
                cls_out: multi lvl [(N,nCls,h,w),(N,nCls,h,w),..]
                pts_init_outs: [(N,np*2,h,w),(N,np*2,h,w),..]  
                pts_refine_outs: [(N,np*2,h,w),(N,np*2,h,w),..]
        '''
        strides = (8,16,32)
        cls_outs = bbox_head_outs['cls_outs']
        pts_refine_outs = bbox_head_outs['pts_refine_outs']

        point_generators = [PointGenerator() for _ in strides]
        mlvl_points = [
            point_generators[i].grid_points(tf.shape(cls_outs[i])[2],tf.shape(cls_outs[i])[3],
                                                 strides[i])
            for i in range(len(strides))
        ]
        # 将（y,x）的表示转换成（x,y）
        pts_refine_outs = [self.points_formate(pts_refine_outs[i]) for i in range(len(strides))]
        result = []
        for i_img in range(cls_outs[0].get_shape().as_list()[0]):
            cls_score_list = [cls_outs[i][i_img,:,:,:] for i in range(len(strides))]
            pts_pred_list = [pts_refine_outs[i][i_img,:,:,:] for i in range(len(strides))]

            mlvl_scores = []
            mlvl_pts = []
            # get result per image
            for i_lvl,(cls_score,pts_pred,points) in enumerate(
                zip(cls_score_list, pts_pred_list, mlvl_points)):

                cls_score = tf.transpose(cls_score,perm=(1,2,0))
                pts_pred = tf.transpose(pts_pred,perm=(1,2,0))

                cls_score = tf.reshape(cls_score,[-1,cfg.DATA.NUM_CATEGORY])
                scores = tf.sigmoid(cls_score)
                pts_pred = tf.reshape(pts_pred,[-1,tf.shape(pts_pred)[-1]/2,2]) # [ns,9,2]

                pts_pos_center = points[:,:2] 
                pts_pos_center =tf.expand_dims(pts_pos_center,1)
                # pts_pos_center = tf.reshape(pts_pos_center,[-1,2])
                # pts 预测的是相对于中心的偏移量
                instance_pts = pts_pred*strides[i_lvl]+pts_pos_center

                mlvl_pts.append(instance_pts)
                mlvl_scores.append(scores)
            mlvl_scores = tf.concat(mlvl_scores,axis=0)
            mlvl_pts = tf.concat(mlvl_pts,axis=0)

            det_boxs,det_pts,det_labels = multiclass_nms(mlvl_pts,mlvl_scores)

            result.append([det_boxs,det_pts,det_labels])
        return result

if __name__ == "__main__":
    import faulthandler; faulthandler.enable()
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    # tf.enable_eager_execution()
    tf.disable_eager_execution()
    # import crash_on_ipy
    # with tf.device('/gpu:0'):
    from config import finalize_configs
    from dataset import register_text

    # from keras import backend as K
    # K.set_session(sess)

    register_text('/home/lupu/27_screenshot/LSVT')
    finalize_configs(True)
    
    model = RepPointsFPNDet()
    image = tf.placeholder(shape=[None,None,3],dtype=tf.float32)
    # image = np.random.rand(640,640,3)*255
    # image = tf.Variable(image,dtype=tf.float32)

    box_outs = model.build_graph(image)
    print(box_outs)
    init = tf.initialize_all_variables()
    # print(box_outs)
    sess = tf.Session()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(box_outs, feed_dict={image: np.ones((700,700,3),dtype=np.int32)})  
        import time
        t0=time.time()
        for i in range(10):
            ss = (600+i*10,600+i*10,3)
            # ss = (700,700,3)
            result = sess.run(box_outs, feed_dict={image: np.random.randint(255,size=ss)})
        print((time.time()-t0)/10)
