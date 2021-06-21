import os
import argparse
import model
import torch
from configparser import ConfigParser
from operator import attrgetter
from utils.watcher import ActivationWatcher
from model.lenet import LeNet

num_classes = {"imagenet": 1000, "cifar10": 10, "cifar100": 100}
parser = argparse.ArgumentParser()
# setup for model
parser.add_argument('--model', default='', help='model', type=str)
parser.add_argument('--config', default='', type=str,
                    help='use configure for words and blocksize')
parser.add_argument('--dataset', default='imagenet', type=str,
                    help="dataset")
parser.add_argument('--conv-word', default=0, help='words update conv layer', type=int)
parser.add_argument('--conv-block', default=1.0, help='blocksize update conv layer', type=float)
parser.add_argument('--fc-word', default=0, help='words update fc layer', type=int)
parser.add_argument('--fc-block', default=1.0, help='blocksize update fc layer', type=float)

parser.add_argument('--init', default=0, help='initialize', type=int)
parser.add_argument('--conv-word-init', default=0, help='conv words initialize', type=int)
parser.add_argument('--conv-block-init', default=0, help='conv blocksize initialize', type=int)
parser.add_argument('--fc-word-init', default=0, help='fc words initialize', type=int)
parser.add_argument('--fc-block-init', default=0, help='fc blocksize initialize', type=int)



def compute_rate(layers, rate=0.95):
    layer_rate = {}
    weight = {}
    total = 0
    for layer in layers:
        weight[layer] = attrgetter(layer+ '.weight.data')(model).detach().numel()
        total += weight[layer]
    for layer in layers:
        layer_rate[layer] = pow(weight[layer]/total, 2)
    return layer_rate

if __name__=='__main__':
    args = parser.parse_args()
    blockconfig = ConfigParser()
    wordconfig = ConfigParser()
    blockconfig.read(os.path.join(args.config, 'block.config'), encoding='UTF-8')
    wordconfig.read(os.path.join(args.config, 'word.config'), encoding='UTF-8')
    model = model.__dict__[args.model](pretrained=(args.dataset == 'imagenet'), num_classes=num_classes[args.dataset])
    watcher = ActivationWatcher(model)
    layers = [layer for layer in watcher.layers[13:]]

    #layer_rate = compute_rate(layers=layers)
    for layer in layers:
        #update the words 
        if(layer in wordconfig[args.model]):
            current_words = int(wordconfig[args.model][layer])
            if 'conv' in layer or 'features' in layer:
                if args.init and args.conv_word_init:
                    wordconfig[args.model][layer] = str(args.conv_word_init)
                else:
                    wordconfig[args.model][layer] = str(int(current_words - args.conv_word)) #
            if 'fc' in layer or 'classifier' in layer or 'linear' in layer:
                if args.init and args.fc_word_init:
                    wordconfig[args.model][layer] = str(args.fc_word_init)
                else:
                    wordconfig[args.model][layer] = str(int(current_words - args.fc_word)) #
        else: # initialization
            wordconfig.set(args.model, layer, str(0))
        with open(os.path.join(args.config, 'word.config'), 'w', encoding='utf-8') as wordfile:
            wordconfig.write(wordfile)
            
        #update the blocks
        if(layer in blockconfig[args.model]):
            current_block = int(blockconfig[args.model][layer])
            if 'conv' in layer or 'features' in layer:
                if args.init and args.conv_block_init:
                    blockconfig[args.model][layer] = str(args.conv_block_init)
                else:
                    blockconfig[args.model][layer] = str(int(current_block * args.conv_block))
            if 'fc' in layer or 'classifier' in layer or 'linear' in layer:
                if args.init and args.fc_block_init:
                    blockconfig[args.model][layer] = str(args.fc_block_init)
                else:
                    blockconfig[args.model][layer] = str(int(current_block * args.fc_block))
        else: # initialization
            blockconfig.set(args.model, layer, str(0))
        with open(os.path.join(args.config, 'block.config'), 'w', encoding='utf-8') as blockfile:
                blockconfig.write(blockfile)
                
    wordfile.close()
    blockfile.close()
