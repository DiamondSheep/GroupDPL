#!/bin/sh
model=resnet18 # resnet18 | vgg16_bn | resnet50
dataset=imagenet
config=/home/gaoyangcheng/dev/Quantize_DPL/config/${model}/
log=/home/gaoyangcheng/dev/Quantize_DPL/log/log_`date +%Y%m%d`.log
touch ${log}
> ${log}
python Quantize_DL.py --dataset ${dataset} --model ${model} \
		--layer all --device cuda --config ${config}  | tee -a ${log}
#>> /gaoyangcheng/Quantize_DPL/log_DL.log
