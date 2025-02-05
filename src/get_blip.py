
import os 
import sys
from BLIP.models.blip import blip_feature_extractor, blip_decoder

# from S3D.s3dg import S3D

def load_blip(filename,decoder=False,vit='base') :

    if decoder: 
        return blip_decoder(filename,vit=vit)
    
    model=blip_feature_extractor(filename,vit=vit)

    return model