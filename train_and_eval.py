import argparse
import torch
from configparser import ConfigParser
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import os

import model
from utils.utils import load_dataset, Visual_data

parser = argparse.ArgumentParser(description='Train and evaluate models in pytorch')
parser.add_argument('--data-path', default='/mnt/dataset/', 
                    help='Path to dataset', type=str)
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset to train or evaluate')
parser.add_argument('--learning-rate', default=0.1, type=float, metavar='LR', 
                    help='initial learning rate(default: 0.1)')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W', 
                    help='weight decay (default: 5e-4)')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--net', default='resnet20_dc', type=str, metavar='NET',
                    help='net type')
parser.add_argument('--num-workers', default=4, type=int, metavar='Wks',
                    help='number of workers')

parser.add_argument('--config', default='./config/', type=str,
                    help='use configure for words and blocksize')
parser.add_argument('--path-save', default='./model_path', type=str,
                    help='Path to save model')
parser.add_argument('--path-model', default='./model_path', type=str,
                    help='Path to model pretrained')
parser.add_argument('--eval', default=True, type=bool,
                    help='evaluate model')

def train(net, trainloader, criterion, optimizer):
    net.train()
    for i, data in enumerate(trainloader, start=0):
        if torch.cuda.is_available():
            input, label = data[0].cuda(), data[1].cuda()
        else:
            input, label = data[0], data[1]
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(net, test_loader, criterion, showFlag=False):
    net.eval()
    top5 = AverageMeter()
    top1 = AverageMeter()
    #losses = AverageMeter()
    
    for i, data in enumerate(test_loader):
        if torch.cuda.is_available():
            input, label = data[0].cuda(), data[1].cuda()
        else:
            input, label = data[0], data[1]
        output = net(input)
        #loss = criterion(output, label)
        prec1, prec5 = accuracy(output.data, label.data, topk=(1, 5))
        
        #losses.update(prec1.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        if showFlag:
            print('Test: [{0}/{1}]\t'
            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            i, len(test_loader), 
            top1=top1, top5=top5))
    return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    wordconfig = ConfigParser()
    wordconfig.read(os.path.join(args.config, 'resnet20/word.config'), encoding='UTF-8')
    word_list = []
    for w in wordconfig['resnet20']:
        word_list.append(int(wordconfig['resnet20'][w]))
    #load dataset
    print('Dataset: {}'.format(args.dataset))
    train_loader, test_loader, num_classes = load_dataset(args.dataset,
                                    args.data_path, args.batch_size, args.num_workers)
    #load model
    net_name = args.net
    print('Model: {}'.format(net_name))
    if 'dc' in net_name:
        net = model.__dict__[net_name](wordconfig = word_list, num_classes=num_classes)
    else:
        net = model.__dict__[net_name](pretrained=args.eval, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    #device
    if torch.cuda.is_available():
        net = net.cuda()
        criterion = criterion.cuda()
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    
    if args.eval: #evaluate
        print('Evaluating started...')
        if args.path_model:
            PATH = os.path.join(args.path_model, '{}-best.pth'.format(net_name))
            net.load_state_dict(torch.load(PATH))
        acc = evaluate(net, test_loader, criterion)
        print('Top1: {}, Top5: {}'.format(acc[0], acc[1]))
    
    else: #train
        print('Hyperparameters:\nlr: {}, momentum: {}, weight_decay: {}'.format(args.learning_rate, args.momentum, args.weight_decay))
        print('Training started...')
        print('Epochs: {}'.format(args.epochs))\
        
        best_acc = 0.0
        time_start = time.time()
        vd = Visual_data()
        for epoch in range(args.epochs):
            loss = train(net, train_loader, criterion, optimizer)
            acc = evaluate(net, test_loader, criterion)
            print('epoch: {} , top1: {:.2f}, top5: {:.2f}'.format(epoch, acc[0], acc[1]))
            vd.update(acc[0], loss)
            if epoch > 10 and best_acc < acc[0]:
                best_acc = acc[0]
                torch.save(net.state_dict(), os.path.join(args.path_save, 
                            '{}-best.pth'.format(net_name)))
        time_end = time.time()
        print('Training finished.')
        print('TrainingTime: {:.3f}s.'.format(time_end - time_start))

        if not os.path.exists(args.path_save):
            os.mkdir(args.path_save)
        PATH = os.path.join(args.path_save, '{}-{}.pth'.format(net_name, args.epochs))
        print('Path: ' + PATH)
        torch.save(net.state_dict(), PATH)
        print('Path Saved.')
        vd.plot_acc(net_name)
        vd.plot_loss(net_name)

