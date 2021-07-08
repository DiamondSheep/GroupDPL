import time
import os
import argparse
from typing import Dict
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from operator import attrgetter, concat, mod
from configparser import ConfigParser

from utils.watcher import ActivationWatcher
from utils.reshape import reshape_weight, reshape_back_weight
from utils.utils import load_dataset, weight_visual, SizeComputation
from train_and_eval import evaluate
import model
import DPL

parser = argparse.ArgumentParser(description='Dicts')
# setup for model
parser.add_argument('--data-path', default='/mnt/dataset', 
                    help='Path to dataset', type=str)
parser.add_argument('--dataset', default='imagenet', type=str,
                    help='dataset to train or evaluate')
parser.add_argument('--model', default='resnet18', type=str, metavar='NET',
                    help='net type (default: resnet18)')
parser.add_argument('--model-path', default='./model_path',
                    help='Path of pretrained model to quantize', type=str)
parser.add_argument('--num-workers', default=4, type=int,
                    help='Number of workers for data loading')
parser.add_argument('--batch-size', default=128, type=int,
                    help='batchsize for training and testing')
# setting for DPL quantization
parser.add_argument('--config', default='', type=str,
                    help='use configure for words and blocksize')
#layer
parser.add_argument('--layer', default='all', type=str,
                    help='compress layer: all, conv, fc')
parser.add_argument('--start', default=1, type=int,
                    help='select start layer for model')
# setting for saving and loading compressed model
parser.add_argument('--path-to-save', default='',
                    help='Path to save compressed layer weight', type=str)
parser.add_argument('--path-to-load', default='',
                    help='Path to load compressed layer weight', type=str) 
# setting for showing and devices
parser.add_argument('--seed', default=0, type=int,
                    help='seed for random initialization')
parser.add_argument('--pretest', default=False, type=bool,
                    help='evaluate the model before compression')
parser.add_argument('--show', default=False, type=bool,
                    help='Show processing information')
parser.add_argument('--device', default='cuda', type=str,
                    help='Device to decompose filters ( cuda | cpu )')
parser.add_argument('--check', default=False, type=bool, 
                    help='check the accuracy loss')

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # config for bloks and words TODO
    blockconfig = ConfigParser()
    wordconfig = ConfigParser()
    groupconfig = ConfigParser()
    blockconfig.read(os.path.join(args.config, 'block.config'), encoding='UTF-8')
    wordconfig.read(os.path.join(args.config, 'word.config'), encoding='UTF-8')
    groupconfig.read(os.path.join(args.config, 'group.config'), encoding='UTF-8')
    word_list = []
    for word in wordconfig[args.model]:
        word_list.append(int(wordconfig[args.model][word]))

    #load dataset
    print('---------------Dataset: {}--------------'.format(args.dataset))
    train_loader, test_loader, num_classes = load_dataset(args.dataset, args.data_path, args.batch_size, args.num_workers)
    
    #load model 
    teacher_model_name = args.model
    student_model_name = args.model+'_dc'
    teacher_model = model.__dict__[teacher_model_name](pretrained=(args.dataset == 'imagenet'), num_classes=num_classes)
    student_model = model.__dict__[student_model_name](wordlist=word_list, num_classes=num_classes)
    
    #load local pretrained model path
    if args.dataset == 'imagenet' and args.model_path:
        teacher_model.load(os.path.join(args.model_path, args.model))
    
    teacher_model.eval()
    criterion = nn.CrossEntropyLoss()

    #device
    if torch.cuda.is_available():
        student_model = student_model.cuda()
        teacher_model = teacher_model.cuda()
        cudnn.benchmark = True
        criterion = criterion.cuda()
    
    #some variables for compute compression time and model size
    model_size = SizeComputation(teacher_model)
    time_compress = 0.0
    pre_loss = 0.0

    #evaluating before compression
    if args.show and args.pretest:
        top_1_before, top_5_before = evaluate(teacher_model, test_loader, criterion, showFlag=False)
        print('\nTop1 before quantization: {:.6f}, Top5 before quantization: {:.6f}\n'.format(top_1_before, top_5_before))
        print('Size of uncompressed model : {:.4f}MB.\n'.format(model_size.uncompressed_size))

    #load layers from model
    teacher_watcher = ActivationWatcher(teacher_model)
    teacher_layers = [layer for layer in teacher_watcher.layers[args.start:]]
    student_watcher = ActivationWatcher(student_model)
    student_layers = [layer for layer in student_watcher.layers[args.start:]]
    print('layer number: {}'.format(len(teacher_layers)))
    
    compressed_layer_list = []
    for layer in teacher_layers:
        teacher_weight = attrgetter(layer+'.weight.data')(teacher_model).detach()
        uncompressed_layer_size = teacher_weight.numel() * 4 / 1024 / 1024
        model_size.update_other(uncompressed_layer_size)
        
        #initialization
        layer_size = 0.0
        is_conv = len(teacher_weight.shape) == 4
        if args.show:
            print('------ layer: {} ------'.format(layer))
            print('size of uncompressed layer: {:.4f}MB'.format(uncompressed_layer_size))
            print('shape: {}'.format(teacher_weight.shape))
        #setting words for layers
        if is_conv:
            if args.layer == 'fc': # compress conv layer only
                model_size.update_reconstruct(uncompressed_layer_size)
                continue
            out_features, in_features, k, _ = teacher_weight.size()
        else:
            if args.layer == 'conv': # compress fc layer only
                model_size.update_reconstruct(uncompressed_layer_size)
                continue
            out_features, in_features = teacher_weight.size()
            k = 1
        
        Decom = DPL.Decomposition()
        Decom.decompose(teacher_weight, k=k, n_word = int(wordconfig[args.model][layer]))
        model_size.update_reconstruct(Decom.layer_size)
        
        compressed_layer_list.append(Decom.DictMat)
        compressed_layer_list.append(Decom.CoefMat)
    assert len(compressed_layer_list) == len(student_layers)
    

    for i, layer in enumerate(student_layers):
        weight = compressed_layer_list[i]
        print(layer)
        print('compressed layer : {}'.format(weight.shape))
        print('target layer :     {}\n'.format(attrgetter(layer+'.weight.data')(student_model).shape))
        assert attrgetter(layer+'.weight.data')(student_model).shape == weight.shape
        attrgetter(layer + '.weight')(student_model).data = weight
    
    top_1, top_5 = evaluate(student_model, test_loader, criterion, showFlag=args.show)
    print('Top1 after quantization: {:.3f}, Top5 after quantization: {:.3f}\n'.format(top_1, top_5))

"""
 I wanna rearrange the dictionary and the coefficients learned by DPL to 
 3x3 convolutional layer and 1x1 convolutional layer respectively. 
 But I failed. 
                                                        --2021/7/8 15:02
"""