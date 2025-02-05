import os 
import sys 
sys.path.insert(1,os.getcwd())


import torch
import torch.nn.functional as F
import time
import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import AutoTokenizer,BlipProcessor,CLIPProcessor
from tqdm import tqdm
from typing import Optional
import numpy as np
import math
import inspect
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn 

from loader.msrvtt_loader import loader_msrvtt
from loader.vatex_loader import loader_vatex
from src.modeling import FullModel
from util_config import ConfigFullModel
from util import  prepare_for_blip,get_logger,get_lr_print
from loader.msvd_loader import loader_msvd
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
import torch.multiprocessing as mp
from src.optimization import BertAdam,cosine_lr_schedule,step_lr_schedule,warmup_lr_schedule




ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available()
    init_process_group(backend='gloo',init_method="env://?use_libuv=False")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 

    dist.barrier()
    
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


## Set the seed
torch.manual_seed(2448)
if torch.cuda.is_available() :
    torch.cuda.manual_seed(2448)


def transform() :
    from torchvision.transforms import transforms
    return  transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))




def uniform_sample_for_vit(video, num_frames=8):
   
    total_frames = video.shape[1]
    if total_frames <= num_frames:
        # If we have fewer frames than required, duplicate frames
        return F.interpolate(video.unsqueeze(0), size=(num_frames, video.shape[2], video.shape[3]), 
                             mode='nearest').squeeze(0)
    else:
        indices = torch.linspace(0, total_frames - 1, num_frames).long()
        return video[:,indices,...]


def prep_optimizer(model):



    # whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention, nn.Conv2d)
    # blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
    # decay=set()
    # no_decay=set()
    
    # pretrained_modules=[
    #     "video_encoder.vision_model.conv1",
    #     "video_encoder.vision_model.class_embedding",
    #     "video_encoder.vision_model.positional_embedding",
    #     "video_encoder.vision_model.ln_pre",
    #     "video_encoder.vision_model.transformer",
    #     "video_encoder.vision_model.ln_post",
    #     "video_encoder.vision_model.proj",
    #     "text_decoder.bert.embeddings.word_embeddings",
    #     "text_decoder.cls.predictions.decoder.",
    #     "text_decoder"
    # ]
    
    # for mn, m in model.named_modules():
    #     for pn, p in m.named_parameters():
    #         fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
    #         if fpn.startswith('module.text_decoder.cls.predictions.decoder') :
    #             continue

    #         if any(fpn.startswith(p_fpn) for p_fpn in pretrained_modules):  # pretrained
    #             no_decay.add(fpn)
    #         elif pn.endswith("bias"):
    #             no_decay.add(fpn)
    #         elif pn.endswith("proj") or pn.endswith("projection"):
    #             decay.add(fpn)
    #         elif fpn.endswith("embedding"):
    #             no_decay.add(fpn)
    #         elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
    #             decay.add(fpn)
    #         elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
    #             no_decay.add(fpn)
    
    
    
    # param_dict = {pn: p for pn, p in model.named_parameters()}

    
    # pretrained_no_decay = [pn for pn in sorted(list(no_decay)) if
    #                         any(pn.startswith(p_pn) for p_pn in pretrained_modules)]
    # not_pretrained_no_decay = [pn for pn in sorted(list(no_decay)) if
    #                                not any(pn.startswith(p_pn) for p_pn in pretrained_modules)]
    
    
    # decay_param = [param_dict[pn] for pn in sorted(list(decay))]
    # no_decay_pretrained_param = [param_dict[pn] for pn in sorted(list(pretrained_no_decay))]
    # no_decay_not_pretrained_param = [param_dict[pn] for pn in sorted(list(not_pretrained_no_decay))]
    
    # optimizer_grouped_parameters = [
    #     {"params": decay_param, "weight_decay" : 0.01, "lr": 6e-4},
    #     {"params": no_decay_pretrained_param, "weight_decay": 0.0, "lr": 2e-6},
    #     {"params": no_decay_not_pretrained_param, "weight_decay": 0.0}
    # ]
    # scheduler=None
    # optimizer=BertAdam(optimizer_grouped_parameters,lr=6e-4,warmup=warmup_time,
    #                    schedule='warmup_linear',t_total=num_train_steps,
    #                    weight_decay=0.01,max_grad_norm=1.0)

    
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.05)
   
    return optimizer



    

@torch.no_grad()
def evaluate(model, data_loader, config,  logger) :
    model.eval()
    tqdm_object=tqdm(data_loader)



    for i,data in enumerate(tqdm_object) :
   
        batch={k:v for k,v in data.items()}

        
        videos=batch['pixel_values']
        videos=transform()(uniform_sample_for_vit(videos,num_frames=config.n_frame)).to(config.device)
     

        video_caption=batch['input_ids'].to(config.device)
        video_caption_attention_mask=batch['attention_mask'].to(config.device)


        audio_attention_mask=batch["audio_caption_attention_mask"].to(config.device)
        audio_ids=batch['audio_caption_ids'].to(config.device)
        

        with torch.autocast(device_type=str(config.device), dtype=torch.bfloat16 ) :
                loss, *other=model(pixel_values=videos,
                    audio_caption_ids=audio_ids.squeeze(),
                    attention_mask=audio_attention_mask.squeeze(),
                    decoder_input_ids=video_caption.squeeze(),
                    decoder_attention_mask=video_caption_attention_mask.squeeze(),
                    labels=video_caption.squeeze(),
                    output_attentions=True,
                    )


        loss=loss/config.grad_accum_steps
        logger.info("Iter: [{0}/{1}], Loss_VAL: {loss:.4f}".format(
                            
            i+ 1,
            len(data_loader),
            loss=loss.detach().item(), 
        ))


    model.train()
    return loss.item()






def train(model, data_loader, optimizer, epoch, config, logger,tokenizer,scheduler=None,file=None,alpha=1.0) : 
    model.train()
    tqdm_object=tqdm(data_loader)


    for i,data in enumerate(tqdm_object) :
   
        batch={k:v for k,v in data.items()}

        videos=batch['pixel_values']
        videos=uniform_sample_for_vit(videos,num_frames=config.n_frame)
        videos=transform()(videos)
        videos=videos.cuda(device)
        


        video_caption=batch['input_ids'].cuda(device)
        video_caption_attention_mask=batch['attention_mask'].cuda(device)

        audio_attention_mask=batch["audio_caption_attention_mask"].cuda(device)
        audio_ids=batch['audio_caption_ids'].cuda(device)
   

        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16 ) :
            loss, *other=model(pixel_values=videos,
                    audio_caption_ids=audio_ids.squeeze(),
                    attention_mask=audio_attention_mask.squeeze(),
                    decoder_input_ids=video_caption.squeeze(),
                    decoder_attention_mask=video_caption_attention_mask.squeeze(),
                    labels=video_caption.squeeze(),
                    )

            loss=loss/config.grad_accum_steps
        loss.backward()


        if ((i+1)%config.grad_accum_steps==0) or ((i+1)==len(data_loader)) : 

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.logits_scale.data.clamp_(0,4.6052)

            if scheduler is not None : 
                scheduler.step()
            optimizer.zero_grad()

        if i%1==0 and master_process: 
            tqdm_object.set_postfix(train_loss=loss.item(),lr=get_lr_print(optimizer))

            logger.info("Epoch: [{0}/{1}], lr: {lr:.2e}, Loss: {loss:.4f}".format(   
            epoch + 1,
            config.max_epoch,
            lr=get_lr_print(optimizer),
            loss=loss.item(), 
        ))

        if i%2==0 and master_process:
            print('Starting Generation ....... \n')

            print('Audio Caption',tokenizer.batch_decode(batch['audio_caption_ids'][0],skip_special_tokens=True))
            print('Video Caption',tokenizer.batch_decode(batch['input_ids'][0],skip_special_tokens=True))

            with torch.no_grad() :
                idx=model.generate(videos[0].unsqueeze(0),audio_ids[0],audio_attention_mask[0])

            print(batch['video_id'][0])
            text=tokenizer.batch_decode(idx)
            print(text)
            
            with open(file,'a',encoding='utf-8') as f :
                f.write(str(epoch) + ' ')
                f.write(batch['video_id'][0]+  ' ')
                f.write(text[0] + '\n')
        

    
    return loss.item()



def main(config,out_dir, is_checkpoint=False, do_evaluate : Optional[bool]=True,file=None) :

    name="bert-base-uncased"
    tokenizer=AutoTokenizer.from_pretrained(name)

    
    model=FullModel(config_model=config)
    model=model.cuda(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model


    config.max_epoch=20
    config.device = torch.cuda.set_device(device)

    data_loader=loader_vatex(tokenizer,"trainval",config=config,ddp=ddp,rank=ddp_rank,world_size=ddp_world_size)
    num_training_steps =  int((len(data_loader) + config.grad_accum_steps -1)/config.grad_accum_steps) * config.max_epoch
    num_warmup_steps =0.05 * num_training_steps
    
    config.warmup_steps=num_warmup_steps

    optimizer=prep_optimizer(model)



    name_log='log_exp20'
    dir_name=f'./Modules\\output\\Caption\\{name_log}.txt'
    logger=get_logger(dir_name)
    

    start_time=time.time()
    for epoch in range(0, config.max_epoch) :
        data_loader.sampler.set_epoch(epoch)

        cosine_lr_schedule(optimizer,epoch,config.max_epoch,config.init_lr,min_lr=0)
        loss=train(raw_model,data_loader,optimizer,epoch,config,logger,tokenizer,scheduler=None,file=file,alpha=1.0)


        if do_evaluate :
            val_data_loader=loader_msvd(tokenizer, "val",config=config)
            evaluate(raw_model, val_data_loader,config,logger)
        if epoch %2==0 and master_process: 
            checkpoint={
                "model" : raw_model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "epoch " :epoch,
                "loss ": loss
            }

            torch.save(checkpoint, out_dir + f"VATEX_{epoch}.pt")
    
    destroy_process_group()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 



if __name__=="__main__": 

    config=ConfigFullModel()
    output_dir="./Modules/output/Caption/"
    file ='Predictions.txt'

    main(config,output_dir,file=file,do_evaluate=False)





        
   

    





    



       