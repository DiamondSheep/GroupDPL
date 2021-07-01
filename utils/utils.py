import os
import torch
from torchvision.utils import save_image

import utils.get_cifar100
import utils.get_cifar10
import utils.get_imagenet

def visual(input_mat, model, layer, mode):
    maxval = torch.max(input_mat)
    minval = torch.min(input_mat)
    input_mat = input_mat / (maxval - minval) * 255
    save_image(input_mat, './visual/{}/mat_{}_{}.png'.format(model, layer, mode))

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