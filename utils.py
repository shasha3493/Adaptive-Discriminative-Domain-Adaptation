# Importing Relevant Libraries 

import os
import shutil
import torch


def save(log_dir, state_dict, is_best):

    if is_best:
        # best model is saved by the name best_model.pt
        best_model_path = os.path.join(log_dir, 'best_model.pt')
        torch.save(state_dict, best_model_path)



class AverageMeter(object):
    """
    Computes and stores the average and current value
    
    """

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
