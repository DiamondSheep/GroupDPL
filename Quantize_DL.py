import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from operator import attrgetter
from configparser import ConfigParser

from utils.watcher import ActivationWatcher
from utils.reshape import reshape_weight, reshape_back_weight
from utils.utils import compute_size
import utils.get_cifar100
import utils.get_cifar10
import utils.get_imagenet
from train_and_eval import evaluate
import model
import DL

parser = argparse.ArgumentParser(description='Dicts vs kmeans')
# setup for model
parser.add_argument('--data-path', default='/mnt/dataset', 
                    help='Path to dataset', type=str)
parser.add_argument('--dataset', default='imagenet', type=str,
                    help='dataset to train or evaluate')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--model', default='resnet18', type=str, metavar='NET',
                    help='net type (default: resnet18)')
parser.add_argument('--model-path', default='/home/gaoyangcheng/dev/Quantize_DPL/model_path',
                    help='Path of pretrained model to quantize', type=str)
parser.add_argument('--n-workers', default=2, type=int,
                    help='Number of workers for data loading')
parser.add_argument('--batch-size', default=256, type=int,
                    help='Batch size')
parser.add_argument('--distributed', default=False, type=bool,
                    help='For multiprocessing distributed')
# setting for DPL quantization
# setting for DPL quantization
parser.add_argument('--config', default='', type=str,
                    help='use configure for words and blocksize')
#layer
parser.add_argument('--layer', default='all', type=str,
                    help='compress layer: all, conv, fc')
# words
parser.add_argument('--n-word-fc', default=256, type=int,
                    help='Number of dictionary words for fc layer')
parser.add_argument('--n-word-conv', default=95, type=int,
                    help='Number of dictionary words for conv layer')
parser.add_argument('--n-word-ds', default=64, type=int,
                    help='Number of dictionary words for downsample layer')   
# block size
parser.add_argument('--block',default=False, type=bool,
                    help='Choose to separate into blocks or not.')
parser.add_argument('--block-size-fc', default=256, type=int,
                    help='Block size for fc layer')
parser.add_argument('--block-size-conv', default=256, type=int,
                    help='Block size for conv layer')
parser.add_argument('--block-size-ds', default=64, type=int,
                    help='Block size for downsample layer(resnet)')
# setting for save and load
parser.add_argument('--path-to-save', default='/home/gaoyangcheng/dev/Quantize_DPL/model_path/',
                    help='Path to save compressed layer weight', type=str)
parser.add_argument('--path-to-load', default='',
                    help='Path to load compressed layer weight', type=str) 
                    # /gaoyangcheng/Quantize_DPL/model_path/
parser.add_argument('--pretest', default=False, type=bool,
                    help='evaluate the model before compression')
parser.add_argument('--show', default=False, type=bool,
                    help='Show processing information')
parser.add_argument('--device', default='cpu', type=str,
                    help='Device to decompose filters')

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    # config for bloks and words
    blockconfig = ConfigParser()
    wordconfig = ConfigParser()
    blockconfig.read(os.path.join(args.config, 'block.config'), encoding='UTF-8')
    wordconfig.read(os.path.join(args.config, 'word.config'), encoding='UTF-8')
    #load dataset
    print('---------------Dataset: {}--------------'.format(args.dataset))
    if args.dataset == 'cifar10':
        train_loader = utils.get_cifar10.get_training_dataloader(data_path=args.data_path, 
                                                                batch_size=args.batch_size, 
                                                                num_workers=args.n_workers)
        test_loader = utils.get_cifar10.get_test_dataloader(data_path=args.data_path, 
                                                            batch_size=args.batch_size, 
                                                            num_workers=args.n_workers)
        num_classes=10
    elif args.dataset == 'cifar100':
        train_loader = utils.get_cifar100.get_training_dataloader(data_path=args.data_path, 
                                                                batch_size=args.batch_size, 
                                                                num_workers=args.n_workers)
        test_loader = utils.get_cifar100.get_test_dataloader(data_path=args.data_path, 
                                                            batch_size=args.batch_size, 
                                                            num_workers=args.n_workers)
        num_classes=100
    elif args.dataset == 'imagenet':
        train_loader = utils.get_imagenet.get_train_dataloader(data_path=args.data_path, 
                                                            batchsize=args.batch_size, 
                                                            num_workers=args.n_workers,
                                                            distributed=args.distributed)
        test_loader = utils.get_imagenet.get_val_dataloader(data_path=args.data_path, 
                                                            batchsize=args.batch_size, 
                                                            num_workers=args.n_workers)
        num_classes = 1000
    #model 
    if 'dc' in args.model:
        model = model.__dict__[args.model](pretrained=(args.dataset == 'imagenet'), num_classes=num_classes)
    else:
        model = model.__dict__[args.model](pretrained=(args.dataset == 'imagenet'), num_classes=num_classes)
    if args.dataset != 'imagenet':
        model.load_state_dict(torch.load(os.path.join(args.model_path, 
                        '{}_eps{}_{}.pth'.format(args.model, args.epochs, args.dataset))))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    #device
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
        criterion = criterion.cuda()
    #load layers from model
    layer_start = 1
    watcher = ActivationWatcher(model)
    layers = [layer for layer in watcher.layers[layer_start: ]]
    # eval uncompressed model
    ''' 
    top_1_before, top_5_before = evaluate(model, test_loader, criterion, 
                                        showFlag=(args.dataset == 'imagenet'))
    '''
    size_uncompressed = compute_size(model)
    size_reconstruct = 0.0
    size_other = size_uncompressed
    time_compress = 0.0
    
    if args.show:
        #print('\nTop1 before quantization: {:.6f}, Top5 after quantization: {:.6f}\n'.format(top_1_before, top_5_before))
        print('Size of uncompressed model : {:.4f}MB.'.format(size_uncompressed))

    exit(0)
    for layer in layers:
        #get weight of layer
        M = attrgetter(layer+ '.weight.data')(model).detach()
        size_uncompressed_layer = M.numel() * 4 / 1024 / 1024
        size_other -= size_uncompressed_layer
        if args.show:
            print('Layer: {}, layer shape:{}, uncompressed layer size: {:.6f}MB. '.format(layer, M.size(), size_uncompressed_layer))
        #initialization
        size_layer = 0.0
        M_dl = []
        is_conv = len(M.shape) == 4
        
        if is_conv:
            if args.layer == 'fc': 
                size_reconstruct += size_uncompressed_layer
                continue
            else:    
                out_features, in_features, k, _ = M.size()
        else:
            if args.layer == 'conv':
                size_reconstruct += size_uncompressed_layer
                continue
            else:
                out_features, in_features = M.size()
                k = 1
        n_blocks = 1 if int(blockconfig[args.model][layer]) == 0 else in_features * k * k // int(blockconfig[args.model][layer])
        n_word = int(wordconfig[args.model][layer])
        
        if len(args.path_to_load) > 0:
            try:
                M = torch.load(os.path.join(args.path_to_load, '{}.pth'.format(layer)))
                attrgetter(layer + '.weight')(model).data = M
                '''
                if args.dataset != 'imagenet':
                    top_1, top_5 = evaluate(model, test_loader, criterion, 
                                            showFlag=(args.dataset == 'imagenet'))
                    print('Top1 after quantization: {:.6f}, Top5 after quantization: {:.6f}'.format(top_1, top_5))
                    print('Loss:{:.4f}\n'.format(top_1_before - top_1))
                '''
                print('Layer already quantized, proceeding to next layer')
                
                continue
            except FileNotFoundError:
                #print('Quantize layer')
                pass
        
        #reshape and chunk weight matrix
        M = reshape_weight(M).to(args.device)
        assert M.size(0) % n_blocks == 0
        M_blocks = M.chunk(n_blocks, dim=0)
        
        if args.show:
            print('layer shape (after reshape): {}'.format(M.shape))

        begin = time.time()
        for M_block in M_blocks:
            dl = DL.DL(Data = M_block, DictSize = n_word)
            dl.Update(iterations=20, showFlag=False)
            M_dl.append(torch.matmul(dl.DictMat, dl.CoefMat))
            block_size = dl.CoefMat.numel() * 2/1024/1024 + dl.DictMat.numel() * 2/1024/1024
            size_layer += block_size
        end = time.time()
        time_cost = end - begin
        time_compress += time_cost
        size_reconstruct += size_layer
        # reconstruct weight and evaluate
        M = torch.cat(M_dl, dim=0).float().to('cuda')
        M = reshape_back_weight(M, k=k, conv=is_conv)
        torch.save(M, os.path.join(args.path_to_save, '{}.pth'.format(layer)))
        attrgetter(layer + '.weight')(model).data = M

        if args.show:
            if args.pretest:
                top_1, top_5 = evaluate(model, test_loader, criterion, showFlag=args.show)
                print('Top1 after quantization: {:.6f}, Top5 after quantization: {:.6f}'.format(top_1, top_5))
                print('Loss:{:.4f}\n'.format(top_1_before - top_1))
            print('Dictionary Words: {}'.format(n_word))
            print('Blocks: {}'.format(n_blocks))
            print('Compressed layer size: {:.6f}MB'.format(size_layer))
            print('Time cost: {:.2f}s'.format(time_cost))

    print('#################result####################')
    print('Compressed model size: {:.4f}MB'.format(size_reconstruct + size_other))
    print('Compress Coef: {:.2f}'.format(size_uncompressed / (size_reconstruct + size_other)))
    print('Compressed time: {:.2f}s.\n'.format(time_compress))
    top_1, top_5 = evaluate(model, test_loader, criterion, showFlag=args.show)
    print('Top1 after quantization: {:.3f}, Top5 after quantization: {:.3f}'.format(top_1, top_5))

