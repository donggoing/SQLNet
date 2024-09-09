# encoding = utf-8
import torch
import torchvision.transforms as transforms
from data.FSC147_384 import FSC147_138, collate_fn       

def build_dataset(root, mode, batch_size=1, distributed=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FSC147_138(root, transform=transform, mode=mode)
    if mode == 'train':
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4, pin_memory=True, sampler=sampler)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True, num_workers=4, pin_memory=True,)
    else:
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, drop_last=False, num_workers=1, pin_memory=True, sampler=sampler)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
        