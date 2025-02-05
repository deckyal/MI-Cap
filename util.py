import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import filters
from skimage import transform as skimage_transform

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def get_lr_print(optimizer) :
    for param in optimizer.param_groups :
        return param["lr"]


def normalize_tensor(t : torch.tensor) :
     b,f,c,h,w=t.size()
     t=t.contiguous().view(b,-1)
     t-=t.min(1,keepdim=True)[0]
     t/=t.max(1,keepdim=True)[0]
     t=t.view(b,f,c,h,w)
     return t



def avg(l : torch.tensor):
    l_avg=[]
    sum_=sum(l)
    count=len(l)
    avg=sum_/count
    l_avg.append(avg)

    return l_avg


def prepare_for_blip(image : torch.tensor):
    image=normalize_tensor(image)*255
    image=image.type(torch.uint8)

    return image


def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger




def getAttMap(img: np.ndarray, attn_map: np.ndarray, blur=True) -> np.ndarray:
    """Visual effect for attention_map

    Args:
        img (_type_): _description_
        attn_map (_type_): _description_
        blur (bool, optional): _description_. Defaults to True.

    Returns:
        The superposed image of the attention map and the img
        
    """
   
    if attn_map.shape[0] != img.shape[0] :
        attn_map=skimage_transform.resize(attn_map,(img.shape[:2]), order=3, mode='constant') 

    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map
