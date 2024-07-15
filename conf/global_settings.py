""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#CANGJIE952 dataset path (python version)
CANGJIE952_TRAIN_PATH = '/home/minh-quan/pytorch-cifar100/data/etl_952_singlechar_size_64/952_train'
CANGJIE952_VAL_PATH = '/home/minh-quan/pytorch-cifar100/data/etl_952_singlechar_size_64/952_val'

# mean and std of cangjie dataset
CANGJIE952_TRAIN_MEAN = 0.20013970136642456
CANGJIE952_TRAIN_STD = 0.38466379046440125
CANGJIE952_VAL_MEAN = 0.20011715590953827
CANGJIE952_VAL_STD = 0.3841699957847595

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200
MILESTONES = [10, 20, 40]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








