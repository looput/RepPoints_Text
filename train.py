#!/usr/bin/env python
# File: train.py
import yaml
import argparse
from dataset.text import register_test,register_text_train

from tensorpack import *
from tensorpack.tfutils import collect_env_info

from dataset import register_coco, register_balloon,register_text
from config import config as cfg
from config import finalize_configs
from data import get_train_dataflow
from eval import EvalCallback
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from modeling.reppoint_detector import RepPointsFPNDet

import logging
logging.getLogger('tensorflow').disabled = True

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

import os
if 'S3_ENDPOINT' in os.environ.keys():
    import moxing as mox
    mox.file.shift('os', 'mox')

if __name__ == '__main__':
    # "spawn/forkserver" is safer than the default "fork" method and
    # produce more deterministic behavior & memory saving
    # However its limitation is you cannot pass a lambda function to subprocesses.
    import multiprocessing as mp
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='Load a model to start training from. It overwrites BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='Log directory. Will remove the old one if already exists.',
                        default='')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py", nargs='+')
    parser.add_argument('--config_file', help="json file store config")

    args = parser.parse_args()
    if args.config_file:
        with open(args.config_file,'r') as f:
            dict_cfg = yaml.load(f)
        cfg.from_dict(dict_cfg)
    if args.config:
        cfg.update_args(args.config)
    register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
    register_balloon(cfg.DATA.BASEDIR)  # add the demo balloon datasets to the registry
    register_text(cfg.DATA.BASEDIR)
    register_text_train(cfg.DATA.BASEDIR)

    # Setup logging ...
    is_horovod = cfg.TRAINER == 'horovod'
    if is_horovod:
        hvd.init()
    if not is_horovod or hvd.rank() == 0:
        log_dir = args.logdir if args.logdir is not '' else cfg.LOG_DIR
        logger.set_logger_dir(log_dir, 'k')
    logger.info("Environment Information:\n" + collect_env_info())

    finalize_configs(is_training=True)

    # Create model
    # MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()
    MODEL = RepPointsFPNDet()

    # Compute the training schedule from the number of GPUs ...
    stepnum = cfg.TRAIN.STEPS_PER_EPOCH
    # warmup is step based, lr is epoch based
    init_lr = cfg.TRAIN.WARMUP_INIT_LR * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
    warmup_schedule = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
    warmup_end_epoch = cfg.TRAIN.WARMUP * 1. / stepnum
    lr_schedule = [(int(warmup_end_epoch + 0.5), cfg.TRAIN.BASE_LR)]

    factor = 8. / cfg.TRAIN.NUM_GPUS
    for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
        mult = 0.1 ** (idx + 1)
        lr_schedule.append(
            (steps * factor // stepnum, cfg.TRAIN.BASE_LR * mult))
    logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule))
    logger.info("LR Schedule (epochs, value): " + str(lr_schedule))
    train_dataflow = get_train_dataflow()
    # This is what's commonly referred to as "epochs"
    total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * 8 / train_dataflow.size()
    logger.info("Total passes of the training set is: {:.5g}".format(total_passes))

    # Create callbacks ...
    callbacks = [
        # EnableCallbackIf(
        #     GraphProfiler(dump_tracing=True, dump_event=True),
        #     lambda self: self.trainer.global_step > 20 and self.trainer.global_step < 30),
        PeriodicCallback(
            ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1,checkpoint_dir=args.logdir),
            every_k_steps=cfg.TRAIN.CHECKPOINT_PERIOD),
            # every_k_epochs=cfg.TRAIN.CHECKPOINT_PERIOD),
        # linear warmup
        ScheduledHyperParamSetter(
            'learning_rate', warmup_schedule, interp='linear', step_based=True),
        ScheduledHyperParamSetter('learning_rate', lr_schedule),
        GPUMemoryTracker(),
        HostMemoryTracker(),
        ThroughputTracker(samples_per_step=cfg.TRAIN.NUM_GPUS),
        EstimatedTimeLeft(median=True),
        SessionRunTimeout(600000),   # 10 minute timeout
        GPUUtilizationTracker()
    ]
    if cfg.TRAIN.EVAL_PERIOD > 0:
        callbacks.extend([
            EvalCallback(dataset, *MODEL.get_inference_tensor_names(), args.logdir)
            for dataset in cfg.DATA.VAL
        ])
    
    if 'S3_ENDPOINT' in os.environ.keys():
        from utils.obs_sync import Sync
        callbacks.append(PeriodicCallback(
            Sync(args.logdir,'s3://bucket-ocr-beijing4/lupu/log/'+args.logdir.split('/')[-1]),
            every_k_steps=cfg.TRAIN.CHECKPOINT_PERIOD)
        )

    if is_horovod and hvd.rank() > 0:
        session_init = None
    else:
        if args.load:
            # ignore mismatched values, so you can `--load` a model for fine-tuning
            session_init = SmartInit(args.load, ignore_mismatch=True)
        else:
            session_init = SmartInit(cfg.BACKBONE.WEIGHTS)

    traincfg = TrainConfig(
    # traincfg = AutoResumeTrainConfig(
        model=MODEL,
        data=QueueInput(train_dataflow),
        callbacks=callbacks,
        steps_per_epoch=stepnum,
        max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
        session_init=session_init,
        starting_epoch=cfg.TRAIN.STARTING_EPOCH
    )

    if is_horovod:
        trainer = HorovodTrainer(average=False)
    else:
        trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False)
    launch_train_with_config(traincfg, trainer)
