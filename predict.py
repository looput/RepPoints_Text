#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from sys import exc_info
import yaml

from dataset.text import register_text_train,register_test
import itertools
import numpy as np
import os
import shutil
import tensorflow as tf
import cv2
import tqdm
import glob

import tensorpack.utils.viz as tpviz
from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter
from tensorpack.utils import fs, logger

from dataset import DatasetRegistry, register_coco, register_balloon,register_text
from config import config as cfg
from config import finalize_configs
from data import get_eval_dataflow, get_train_dataflow
from eval import DetectionResult, multithread_predict_dataflow, predict_image
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from modeling.reppoint_detector import RepPointsFPNDet
from viz import (
    draw_annotation, draw_final_outputs, draw_gts, draw_predictions,
    draw_proposal_recall, draw_final_outputs_blackwhite)


def do_visualize(model, model_path, nr_visualize=100, output_dir='output'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_train_dataflow()
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=model,
        session_init=SmartInit(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_{}_proposals/boxes'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'generate_{}_proposals/scores'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'fastrcnn_all_scores',
            'output/boxes',
            'output/scores',
            'output/labels',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df), nr_visualize):
            img, gt_boxes, gt_labels = dp['image'], dp['gt_boxes'], dp['gt_labels']

            rpn_boxes, rpn_scores, all_scores, \
                final_boxes, final_scores, final_labels = pred(img, gt_boxes, gt_labels)

            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels)
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_scores[good_proposals_ind])

            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_scores, final_labels,
                           [None] * len(final_labels))]
            final_viz = draw_final_outputs(img, results)

            viz = tpviz.stack_patches([
                gt_viz, proposal_viz,
                score_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()


def do_evaluate(pred_config, output_file):
    num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
    graph_funcs = MultiTowerOfflinePredictor(
        pred_config, list(range(num_tower))).get_predictors()

    res_path = os.path.join(os.path.split(output_file)[0],'result.txt')
    f = open(res_path,'w')
    final_res = {'tp':0, 'fp':0, 'npos':0}
    for dataset in cfg.DATA.VAL:
        logger.info("Evaluating {} ...".format(dataset))
        dataflows = [
            get_eval_dataflow(dataset, shard=k, num_shards=num_tower)
            for k in range(num_tower)]
        all_results = multithread_predict_dataflow(dataflows, graph_funcs)
        output = output_file + '-' + dataset
        name = DatasetRegistry.get_metadata(dataset,'dataset_names')
        print(name)
        res = DatasetRegistry.get(dataset).eval_inference_results(all_results, output)
        p,r,h = res['precision'],res['recall'],res['hmean']
        m_iou = 0. if 'sum_iou' not in res.keys() else res['sum_iou']/res['c_iou']
        extra_info = ''
        for k in res['statictis'].keys():
            m =sum(res['statictis'][k])/len(res['statictis'][k])
            extra_info+= f'{m:0.4},'

        f.write(f'{name}, {p:0.4}, {r:0.4}, {h:0.4}, {extra_info}\r\n')
        for k in final_res.keys():
            final_res[k]+=res[k]
        print(f'{name}, {p:0.4}, {r:0.4}, {h:0.4}, {extra_info}')
    
    tp,fp,npos = final_res['tp'],final_res['fp'],final_res['npos']
    precision = tp / (tp + fp)
    recall = tp / npos
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    print('----------')
    print('Final res, P,R,F ','%.4f %.4f %.4f'%(precision,recall,hmean))
    f.write(f'total_res, {precision:0.4}, {recall:0.4}, {hmean:0.4}\r\n')

def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    # from common import CusRotation
    # rotator = CusRotation(2,(0.5,0.5),border=cv2.BORDER_CONSTANT,border_value=[123.675, 116.28, 103.53])
    # img = rotator.augment(img)
    import time
    results = predict_image(img, pred_func,time.time())
    if cfg.MODE_MASK:
        final = draw_final_outputs_blackwhite(img, results)
    else:
        final = draw_final_outputs(img, results)
    # draw gt
    try:
        final = draw_gts(final,os.path.splitext(img_pth)[0]+'.txt')
    except:
        pass
    viz =final
    # viz = np.concatenate((img, final), axis=1)
    nm = os.path.basename(input_file).replace('.jpg','.png')
    folder = input_file.split('/')[-2]
    path = f'/home/lupu/shared_space/lupu/TF-LOG/img_log_lite/{folder}'
    if not os.path.isdir(path):
        os.makedirs(path)
    # cv2.imwrite(path+f"/{nm}", viz)
    logger.info("Inference output for {} written to output.png".format(input_file))
    cv2.namedWindow('img_show',0)
    cv2.imshow('img_show',viz)
    key = cv2.waitKey()
    key = chr(key & 0xff)
    if key == 'q':
        import sys
        sys.exit()
    # tpviz.interactive_imshow(viz)


class CusPredictor(OfflinePredictor):
    def __init__(self, config):
        super(CusPredictor,self).__init__(config)
    
    def _do_call(self, dp):
        assert len(dp) == len(self.input_tensors), \
            "{} != {}".format(len(dp), len(self.input_tensors))
        if self.sess is None:
            self.sess = tf.get_default_session()
            assert self.sess is not None, "Predictor isn't called under a default session!"

        if self._callable is None:
            self._callable = self.sess.make_callable(
                fetches=self.output_tensors,
                feed_list=self.input_tensors,
                accept_options=self.ACCEPT_OPTIONS)
        run_metadata = tf.RunMetadata()
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        outs =  self._callable(*dp,options=options,
               run_metadata=run_metadata)
        
        # opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
        #     tf.profiler.ProfileOptionBuilder.time_and_memory())
        #     .with_step(0).with_timeline_output('time_line.json')
        #     # .with_displaying_options(show_name_regexes=['.*reppoints_head.*'])
        #     .build())
        # tf.profiler.profile(
        #     tf.get_default_graph(),
        #     run_meta=run_metadata,
        #     cmd='scope',
        #     options=opts)
        
        # advice = tf.profiler.advise(tf.get_default_graph(), run_meta=run_metadata)
        # tf.profiler.profile(
        #     tf.get_default_graph(),
        #     run_meta=run_metadata,
        #     options=tf.profiler.ProfileOptionBuilder.float_operation())
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/profile_model_architecture.md#caveats
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/profiler/model_analyzer.py
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/python_api.md
        return outs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation.', required=True)
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', help="Run evaluation. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file", nargs='+')
    parser.add_argument('--benchmark', action='store_true', help="Benchmark the speed of the model + postprocessing")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')
    parser.add_argument('--config_file', help="json file store config")
    parser.add_argument('--output-pb', help='Save a model to .pb')
    parser.add_argument('--output-serving', help='Save a model to serving file')

    args = parser.parse_args()
    if args.config_file:
        with open(args.config_file,'r') as f:
            dict_cfg = yaml.load(f)
        cfg.from_dict(dict_cfg)
    if args.config:
        cfg.update_args(args.config)
    register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
    register_balloon(cfg.DATA.BASEDIR)
    register_text(cfg.DATA.BASEDIR)
    register_test(cfg.DATA.BASEDIR)
    register_text_train(cfg.DATA.BASEDIR)

    MODEL = RepPointsFPNDet() if cfg.MODE_FPN else ResNetC4Model()

    if not tf.test.is_gpu_available():
        from tensorflow.python.framework import test_util
        assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
            "Inference requires either GPU support or MKL support!"
    # assert args.load
    finalize_configs(is_training=False)

    if args.predict or args.visualize:
        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

    if args.visualize:
        do_visualize(MODEL, args.load)
    else:
        predcfg = PredictConfig(
            model=MODEL,
            session_init=SmartInit(args.load),
            input_names=MODEL.get_inference_tensor_names()[0],
            output_names=MODEL.get_inference_tensor_names()[1])

        if args.output_pb:
            ModelExporter(predcfg).export_compact(args.output_pb, optimize=False)
        elif args.output_serving:
            ModelExporter(predcfg).export_serving(args.output_serving)

        if args.predict:
            predictor = CusPredictor(predcfg)
            predictor.ACCEPT_OPTIONS=True
            for image_file in args.predict:
                file_lst = glob.glob(image_file)
                for img_pth in file_lst:
                    do_predict(predictor, img_pth)
        elif args.evaluate:
            # assert args.evaluate.endswith('.json'), args.evaluate
            do_evaluate(predcfg, args.evaluate)
        elif args.benchmark:
            df = get_eval_dataflow(cfg.DATA.VAL[0])
            df.reset_state()
            predictor = OfflinePredictor(predcfg)
            for _, img in enumerate(tqdm.tqdm(df, total=len(df), smoothing=0.5)):
                # This includes post-processing time, which is done on CPU and not optimized
                # To exclude it, modify `predict_image`.
                predict_image(img[0], predictor)
