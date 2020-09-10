from torch.utils.data import DataLoader, random_split
import numpy as np
from copy import deepcopy
from dataloader.Dataset import DG_Dataset

def random_split_dataloader(data, data_root, source_domain, target_domain, batch_size, 
                   get_domain_label=False, get_cluster=False, num_workers=4, color_jitter=True, min_scale=0.8, seed=0):
    if data=='VLCS': 
        split_rate = 0.7
    else: 
        split_rate = 0.9
    print(data_root)
    source_train = DG_Dataset(root_dir=data_root, domain=source_domain, split='train',
                                     get_domain_label=False, get_cluster=False, color_jitter=color_jitter, min_scale=min_scale, seed=seed)
    source_val = DG_Dataset(root_dir=data_root, domain=target_domain, split='val',
                              get_domain_label=False, get_cluster=False, color_jitter=color_jitter, min_scale=min_scale, seed=seed)
    
    target_test =  DG_Dataset(root_dir=data_root, domain=target_domain, split='test',
                                   get_domain_label=False, get_cluster=False, seed=seed)
    
    print('Train: {}, Val: {}, Test: {}'.format(len(source_train), len(source_val), len(target_test)))
    
    source_train = DataLoader(source_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    source_val  = DataLoader(source_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    target_test = DataLoader(target_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return source_train, source_val, target_test
