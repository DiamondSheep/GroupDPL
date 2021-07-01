#!/bin/sh
model=resnet50 # resnet18 | vgg16_bn | resnet50 | resnet56 | resnet20
dataset=imagenet # imagenet | cifar10 | cifar100
mkdir ./visual/$model
config=/home/gaoyangcheng/dev/GroupDPL/config/${model}/
log=/home/gaoyangcheng/dev/GroupDPL/log/log_`date +%Y%m%d`.log
touch ${log}
convblocksize=$(expr 72)
convwords=$(expr 16)
fcblocksize=$(expr 32)
fcwords=$(expr 10)
#python update_config.py --config ${config} --init=1 --model=${model} --dataset=${dataset} \
#--conv-word-init=${convwords} --conv-block-init=${convblocksize} 

for j in 1
do
	#echo "convblocksize: ${convblocksize}, fcblocksize: ${fcblocksize}"
	for i in 1
	do
		echo "${model} Quantization $i" | tee -a ${log}
		date | tee -a ${log}
		#echo "convwords: ${convwords}, fcwords: ${fcwords}" | tee -a ${log}
		python main.py --dataset ${dataset} --model ${model} \
		--layer all --device cuda --config ${config} --show True | tee -a ${log} #--pretest True --show True --check True 
		#cat ${config}/word.config | tee -a ${log}
		#python update_config.py --config ${config} --model=${model} --dataset=${dataset} --init=0 --conv-word=3
		#convwords=$(expr ${convwords} + 8)
		#sleep 10
	done
	#python update_config.py --config ${config} --model=${model} --dataset=${dataset} \
	#--init=0 --conv-block=0.5
	#convblocksize=$(expr ${convblocksize} / 2)
	#convwords=$(expr ${convwords} / 3)
	#fcwords=$(expr ${fcwords} + 10)
	#python update_config.py --config ${config} --init=1 --model=${model} --dataset=${dataset} \
	#--conv-word-init=${convwords} --fc-word-init=${fcwords}
	#sleep 120
	echo "----------------------------------------------------------" | tee -a ${log}
done
#python compute_words.py --model resnet18 --config 