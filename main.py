import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from operator import attrgetter, mod
from configparser import ConfigParser

from utils.watcher import ActivationWatcher
from utils.reshape import reshape_weight, reshape_back_weight
from utils.utils import compute_size, load_dataset, visual
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
parser.add_argument('--model-path', default='/home/gaoyangcheng/dev/Quantize_DPL/model_path',
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
    blockconfig.read(os.path.join(args.config, 'block.config'), encoding='UTF-8')
    wordconfig.read(os.path.join(args.config, 'word.config'), encoding='UTF-8')
    #load dataset
    print('---------------Dataset: {}--------------'.format(args.dataset))
    train_loader, test_loader, num_classes = load_dataset(args.dataset,
                                    args.data_path, args.batch_size, args.num_workers)
    #load model 
    model = model.__dict__[args.model](pretrained=(args.dataset == 'imagenet'), num_classes=num_classes)
    if args.dataset != 'imagenet': #load private pretrained model 
        pass
    model.eval()
    criterion = nn.CrossEntropyLoss()

    #device
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
        criterion = criterion.cuda()

    #some variables for compute compression time and model size
    size_uncompressed = compute_size(model)
    size_reconstruct = 0.0
    size_other = size_uncompressed
    time_compress = 0.0
    pre_loss = 0.0

    #evaluating before compression
    if args.show and args.pretest:
        top_1_before, top_5_before = evaluate(model, test_loader, criterion, showFlag=False)
        print('\nTop1 before quantization: {:.6f}, Top5 before quantization: {:.6f}\n'.format(top_1_before, top_5_before))
        print('Size of uncompressed model : {:.4f}MB.\n'.format(size_uncompressed))
    
    #load layers from model
    watcher = ActivationWatcher(model)
    layers = [layer for layer in watcher.layers[args.start:]]
    
    #-------------------------------   compression   ------------------------------    
    for layer in layers:
        #get weight of layer
        M = attrgetter(layer+ '.weight.data')(model).detach()
        size_uncompressed_layer = M.numel() * 4 / 1024 / 1024
        size_other -= size_uncompressed_layer
        #initialization
        size_layer = 0.0
        M_dpl = []
        is_conv = len(M.shape) == 4

        if args.show:
            print('Layer: {}, layer shape: {}, uncompressed layer size: {:.6f}MB. '.format(layer, M.size(), size_uncompressed_layer))
        
        #load compressed model
        if len(args.path_to_load) > 0:
            try:
                M = torch.load(os.path.join(args.path_to_load, '{}.pth'.format(layer)))
                attrgetter(layer + '.weight')(model).data = M
                if args.dataset != 'imagenet':
                    top_1, top_5 = evaluate(model, test_loader, criterion,
                                            showFlag=(args.dataset == 'imagenet'))
                    print('Top1 after quantization: {:.6f}, Top5 after quantization: {:.6f}'.format(top_1, top_5))
                    print('Loss:{:.4f}'.format(top_1_before - top_1))
                if args.show:
                    print('Layer already compressed, proceeding to next layer')
                continue
            except FileNotFoundError:
                pass

        #setting words and blocks for layers
        if is_conv:
            if args.layer == 'fc': # Only compress conv layer
                size_reconstruct += size_uncompressed_layer
                continue
            else:    
                out_features, in_features, k, _ = M.size()
        else:
            if args.layer == 'conv': # Only compress fc layer
                size_reconstruct += size_uncompressed_layer
                continue
            else:
                out_features, in_features = M.size()
                k = 1
        n_blocks = 1 if int(blockconfig[args.model][layer]) == 0 else in_features * k * k // int(blockconfig[args.model][layer])
        n_word = int(wordconfig[args.model][layer])

        #reshape and chunk weight matrix
        M = reshape_weight(M).to(args.device)
        
        if args.model == 'resnet50': 
            if 'conv3' in layer or 'downsample' in layer:
                M = M.t()
                n_blocks = out_features // int(blockconfig[args.model][layer])
        if args.show:
            print('layer shape (after reshape): {}'.format(M.shape))
        assert M.size(0) % n_blocks == 0
        M_blocks = M.chunk(n_blocks, dim=0)

        #__________________________   DPL decomposition   ________________________
        begin = time.time()
        for M_block in M_blocks:
            dpl = DPL.DPL(Data = M_block, DictSize=n_word, tau=0.05, using_cuda=(args.device == 'cuda'))
            dpl.Update(iterations=10, showFlag=False)
            M_dpl.append(torch.matmul(dpl.DictMat, torch.matmul(dpl.P_Mat, dpl.DataMat)))
            block_size = torch.matmul(dpl.P_Mat, dpl.DataMat).numel() * 2/1024/1024 + dpl.DictMat.numel() * 2/1024/1024
            size_layer += block_size

            '''
            # Visualization for the weight 
            visual(M_block, model=args.model, layer=layer, mode ='origin')
            visual(dpl.DictMat, model=args.model, layer=layer, mode='dict')
            visual(dpl.P_Mat, model=args.model, layer=layer, mode='P')
            visual(torch.matmul(dpl.P_Mat, dpl.DataMat), model=args.model, layer=layer, mode='coef')
            visual(torch.matmul(dpl.DictMat, torch.matmul(dpl.P_Mat, dpl.DataMat)),
                                model=args.model, layer=layer, mode='rebuild')
            '''
            
        #_________________________________________________________________________
        
        end = time.time()
        time_cost = end - begin
        time_compress += time_cost
        size_reconstruct += size_layer

        # reconstruct
        M = torch.cat(M_dpl, dim=0).float().to('cuda')
        if args.model == 'resnet50': 
            if 'conv3' in layer or 'downsample' in layer:
                M = M.t()
        M = reshape_back_weight(M, k=k, conv=is_conv)
        attrgetter(layer + '.weight')(model).data = M

        if args.path_to_save:
            torch.save(M, os.path.join(args.path_to_save, '{}.pth'.format(layer)))
        if args.show and args.pretest:
            top_1, top_5 = evaluate(model, test_loader, criterion, showFlag=False)
            print('Top1 after quantization: {:.6f}, Top5 after quantization: {:.6f}'.format(top_1, top_5))
            print('Loss:{:.4f}'.format(top_1_before - top_1))
            
            if args.check: # for tuning the words number manually 
                if (top_1_before - top_1 - pre_loss) > 0.02:
                    print("*************modify the words in {}.**************".format(layer))
                if (top_1_before - top_1 < pre_loss):
                    print("reduce the words!")
                pre_loss = top_1_before - top_1

            print('Dictionary Words:      {}'.format(n_word))
            print('Blocks:                {}'.format(n_blocks))
            print('Compressed layer size: {:.6f}MB'.format(size_layer))
            print('Time cost:             {:.2f}s\n'.format(time_cost))
        #--------------------------------------------------------------------------
    #result print
    print('#################result####################')
    print('Compressed model size: {:.4f}MB'.format(size_reconstruct + size_other))
    print('Compress Coef: {:.2f}'.format(size_uncompressed / (size_reconstruct + size_other)))
    print('Compressed time: {:.2f}s.'.format(time_compress))
    #top_1, top_5 = evaluate(model, test_loader, criterion, showFlag=args.show)
    #print('Top1 after quantization: {:.3f}, Top5 after quantization: {:.3f}\n'.format(top_1, top_5))
