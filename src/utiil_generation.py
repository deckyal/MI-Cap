import torch
import torch.nn as nn 
import torch.nn.functional as F
from transformers import PreTrainedModel,AutoProcessor
from typing import List,Tuple,Optional



def normalize_tensor(t : torch.tensor) :
     b,f,c,h,w=t.size()
     t=t.view(b,-1)
     t-=t.min(1,keepdim=True)[0]
     t/=t.max(1,keepdim=True)[0]
     t=t.view(b,f,c,h,w)
     return t



def prepare_for_blip(image : torch.tensor):
    image=normalize_tensor(image)*255
    image=image.type(torch.uint8)

    return image


def prepare_for_generation(image, image_processor) :
    image=prepare_for_blip(image)
    images=torch.stack([image_processor(image[i],return_tensors="pt").pixel_values for i in range(image.shape[0])])
    return images




def query_generate(Model : List[nn.Module], pixel_values,audio_ids: Optional[torch.tensor] ,image_processor : Optional[AutoProcessor], _hf ) :

    image=prepare_for_generation(pixel_values,image_processor)
    out=torch.zeros(len(Model))
    for model in Model :
        if _hf : 
            text=model.generate(image,max_length=30,num_beams=3, min_length=10, top_p=0.9, repetition_penalty=1.0,do_sample=True)
            out=torch.cat((out,text))

        else :
            text=model.generate(image,audio_ids,top_p=True)
            out=torch.cat((out,text))
    
    return out






