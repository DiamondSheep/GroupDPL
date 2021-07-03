#!/bin/sh
model=resnet20_dc # resnet18 | vgg16_bn | resnet50 | resnet56 | resnet20
dataset=cifar10 # imagenet | cifar10 | cifar100
config=/home/gaoyangcheng/dev/GroupDPL/config/${model}/
log=/home/gaoyangcheng/dev/GroupDPL/log/log_`date +%Y%m%d`.log
touch ${log}

python train_and_eval.py | tee -a ${log}