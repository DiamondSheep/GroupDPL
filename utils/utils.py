import os
import torch
from torchvision.utils import save_image

import utils.get_cifar100
import utils.get_cifar10
import utils.get_imagenet

def weight_visual(input_mat, model, layer, mode):
    maxval = torch.max(input_mat)
    minval = torch.min(input_mat)
    input_mat = input_mat / (maxval - minval) * 255
    save_image(input_mat, './visual/{}/mat_{}_{}.png'.format(model, layer, mode))

class SizeComputation():
    """
    compute compressed model size
    """
    def __init__(self, model):
        self.uncompressed_params = 0
        self.uncompressed_size = 0.0
        self.reconstruct_size = 0.0
        self.compressed_size = 0.0
        for n, p in model.named_parameters():
            self.uncompressed_params += p.numel()
        self.uncompressed_size = self.uncompressed_params * 4 / 1024 / 1024
        self.other_size = self.uncompressed_size
    def update_compressed(self):
        self.compressed_size = self.other_size + self.reconstruct_size

    def update_other(self, uncompressed_layer_size):
        self.other_size -= uncompressed_layer_size
        self.update_compressed()

    def update_reconstruct(self, compressed_layer_size):
        self.reconstruct_size += compressed_layer_size
        self.update_compressed()
        
def compute_size(model):
    """
    Size of model (in MB).
    """

    res = 0
    for n, p in model.named_parameters():
        res += p.numel()

    return res * 4 / 1024 / 1024

def load_dataset(dataset, data_path, batch_size, num_workers, distributed=False):
    if dataset == 'cifar10':
        train_loader = utils.get_cifar10.get_training_dataloader(data_path=data_path, 
                                                                batch_size=batch_size, 
                                                                num_workers=num_workers)
        test_loader = utils.get_cifar10.get_test_dataloader(data_path=data_path, 
                                                            batch_size=batch_size, 
                                                            num_workers=num_workers)
        num_classes=10
    elif dataset == 'cifar100':
        train_loader = utils.get_cifar100.get_training_dataloader(data_path=data_path, 
                                                                batch_size=batch_size, 
                                                                num_workers=num_workers)
        test_loader = utils.get_cifar100.get_test_dataloader(data_path=data_path, 
                                                            batch_size=batch_size, 
                                                            num_workers=num_workers)
        num_classes=100
    elif dataset == 'imagenet':
        # no need for train dataset
        train_loader = utils.get_imagenet.get_train_dataloader(data_path=data_path, 
                                                            batchsize=batch_size, 
                                                            num_workers=num_workers,
                                                            distributed=distributed)
        
        test_loader = utils.get_imagenet.get_val_dataloader(data_path=data_path, 
                                                            batchsize=batch_size, 
                                                            num_workers=num_workers)
        num_classes = 1000
    return train_loader, test_loader, num_classes

import matplotlib.pyplot as plt

class Visual_data(object):
    def __init__(self):
        self.acc_list = []
        self.loss_list = []
        self.epoch = 0

    def update(self, acc, loss):
        self.acc_list.append(acc)
        self.loss_list.append(loss)
        self.epoch += 1
        assert len(self.acc_list)==self.epoch
        assert len(self.loss_list)==self.epoch

    def plot_acc(self, model):
        plt.plot(self.acc_list, 'deepskyblue', marker='o', ls='-')
        plt.ylabel('acc')
        plt.xlabel('epochs')
        #plt.show()
        plt.savefig('./visual/acc_{}.png'.format(model))
        plt.close()
    
    def plot_loss(self, model):
        plt.plot(self.loss_list, c='orange', marker='o', ls='-')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        #plt.show()
        plt.savefig('./visual/loss_{}.png'.format(model))
        plt.close()