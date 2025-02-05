import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional,Tuple
from transformers import AutoTokenizer,BlipForConditionalGeneration
from transformers import BertConfig,BertTokenizer
import numpy as np

from .model_cross import CoModalModule
from .model_temporal import TemporalEncoder
from .model_components import SmallDecoder,VideoEncoder,AudioCaptionModel,BlipDecoder,AudioLMHead
from .model_pooler import AggModule
from .get_blip import load_blip


def init_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        return tokenizer

        
def mlm_loss(logits,labels) :
    
    return F.cross_entropy(logits.view(-1,logits.size(-1)),labels.view(-1),ignore_index=0)



def filter_dict(state_dict, filter_key, remove_prefix):
    return {k.removeprefix(remove_prefix): v for k, v in state_dict.items() if filter_key in k}






class FullModel(nn.Module) :
    def __init__(self,config_model) :
        super().__init__()

        self.config=config_model
        self.video_encoder=VideoEncoder(blip=True,blip_model=config_model.blip_model,n_frame=self.config.n_frame)

        self.audio_text_encoder=AudioCaptionModel(bert=False)
        self.cross_module=CoModalModule(config_cross=config_model.config_cross)
        self.video_proj=nn.Linear(config_model.hidden_size,config_model.embed_dim)
        self.audio_proj=nn.Linear(config_model.hidden_size,config_model.embed_dim)
        


        self.logits_scale=nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.agg_module=AggModule(config_model) 
 
       
        config = BertConfig.from_pretrained("bert-base-uncased", bos_token_id=101, pad_token_id=0, eos_token_id=102)
        config.is_decoder = True
        config.add_cross_attention = True
        config.num_hidden_layers=6
        # config.hidden_dropout_prob = 0.3
        # config.attention_probs_dropout_prob = 0.3
        self.text_decoder = SmallDecoder.from_pretrained('bert-base-uncased', config=config)

        
        # self.text_decoder=load_blip(config_model.blip_model,decoder=True).text_decoder
        # self.text_decoder.hidden_dropout_prob=0.20
        # self.text_decoder.attention_probs_dropout_prob = 0.20
        

        self.grad=None
        self.tokenizer=init_tokenizer()


    
    def contrast_loss(self,cls_video,cls_audio,logits_scale) : 

        video_embd=cls_video/cls_video.norm(p=2,dim=-1,keepdim=True)
        text_embd=cls_audio/cls_audio.norm(p=2,dim=-1,keepdim=True)


        logits_scale=logits_scale.exp()
        logits_per_video=logits_scale*video_embd@text_embd.t()
        logits_per_text=logits_per_video.t()
        

        loss_vt=F.cross_entropy(logits_per_video,torch.arange(len(logits_per_video),device=logits_per_video.device))
        loss_tv=F.cross_entropy(logits_per_text,torch.arange(len(logits_per_text),device=logits_per_text.device))

        loss=(loss_vt+loss_tv)*0.5

        return loss,logits_per_video,logits_per_text
    
    
   
    
    def forward(self,
                pixel_values : torch.tensor, 
                audio_caption_ids : torch.LongTensor,
                attention_mask : Optional[torch.LongTensor]=None,
                decoder_input_ids : Optional[torch.tensor]=None,
                decoder_attention_mask: Optional[torch.LongTensor]=None,
                labels : Optional[torch.LongTensor]=None ,
               targets : Optional[torch.tensor] = None,
                output_attentions : Optional[bool]=True,
                generate_flag : Optional[bool]=False
                ) :
       

        B=pixel_values.shape[0]
        last_hidden_state_video=self.video_encoder(pixel_values)
       
        audio_features=self.audio_text_encoder(audio_caption_ids=audio_caption_ids,
                                               attention_mask=attention_mask)
        
        last_hidden_state_audio=audio_features.last_hidden_state


        last_hidden_state_video_cross, last_hidden_state_audio_cross, cls_video,cls_audio,*attn_tuple=self.cross_module(video_feat=last_hidden_state_video,
                                                                                               audio_feat=last_hidden_state_audio,  
                                                                                               audio_attention_mask=attention_mask     
                                                                                       )
        
        
        loss_contrast,logits_video,logits_text=self.contrast_loss(cls_video,cls_audio,self.logits_scale)
    
        
    
        video_hidden_state=self.agg_module(last_hidden_state_video_cross,last_hidden_state_audio_cross)

        labels = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100) 
        out=self.text_decoder(input_ids=decoder_input_ids,
                                attention_mask=decoder_attention_mask,
                                encoder_hidden_states=video_hidden_state,
                                output_attentions=output_attentions,
                                labels=labels,
                                )
        
        


        loss_model=None
        if labels is not None :
            loss_model=out.loss + loss_contrast
        
        
        outputs=(loss_model,out.logits)
        if output_attentions : 
         outputs+= (attn_tuple,) + (logits_video,logits_text) + (cls_video,cls_audio)

        return  outputs
    
    @torch.no_grad()
    def prepare_for_generation(self,pixel_values,audio_caption_ids,attention_mask) :
        last_hidden_state_video=self.video_encoder(pixel_values)
       

        audio_features=self.audio_text_encoder(audio_caption_ids=audio_caption_ids,
                                               attention_mask=attention_mask)
        last_hidden_state_audio=audio_features.last_hidden_state


        last_hidden_state_video, last_hidden_state_audio,pooled_video, pooled_audio,*attn_tuple=self.cross_module(last_hidden_state_video,
                                                                                               last_hidden_state_audio,
                                                                                               audio_attention_mask=attention_mask)
        video_hidden_state=self.agg_module(last_hidden_state_video,last_hidden_state_audio)


        return video_hidden_state

    def generate(self, videos,audio_caption,audio_attention_mask,input_ids=None, sample=False, num_beams=4,seq=2,
                 temperature=1.0, max_length=30, min_length=5,
                 top_p=0.9, repetition_penalty=0.6,early_stoping=True):
       

        image_embeds=self.prepare_for_generation(pixel_values=videos,
                                                 audio_caption_ids=audio_caption,
                                                 attention_mask=audio_attention_mask)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(videos.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[101, 2]])
                .repeat(image_embeds.size(0), 1)
                .to(image_embeds.device)
            )

        input_ids[:,0] = 101
        input_ids = input_ids[:, :-1] 

        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  temperature=temperature,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            # print('beam search')
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  num_return_sequences=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
      
        return outputs
         


    @torch.no_grad()
    def generate_manual(self, videos, audio_caption,audio_attention_mask,top_k=True,device='cuda:1') :
        
        self.eval()
        b=audio_caption.shape[0]
        if b>1 : 
            videos=videos[0].unsqueeze(0)
            audio_caption_ids=audio_caption[0]
            audio_attention_mask=audio_attention_mask[0]

        idx=torch.LongTensor([101]).view(1,1).to(device)

        if top_k :
            for _ in range(self.config.max_length) :

                with torch.autocast(device_type=str(torch.device("cuda")), dtype=torch.bfloat16):

                    out=self(pixel_values=videos,
                            audio_caption_ids=audio_caption,
                            attention_mask=audio_attention_mask,
                            decoder_input_ids=idx,
                            output_attentions=True,
                            )
                    
                    logits=out[1]
                


                    logits=logits[:,-1,: ]/0.7

                    v, _ = torch.topk(logits, min(self.config.k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                    probs = F.softmax(logits, dim=-1)

                    # sample from the distribution
                    idx_next = torch.multinomial(probs, num_samples=1)

                    
                    if idx[:,-1]==102 :
                        break

                    # append sampled index to the running sequence and continue
                    idx = torch.cat((idx, idx_next), dim=1)
            return idx


    @torch.no_grad()
    def from_pretrained(self,model_type,_self=True) :
        print("loading weights from pretrained Model: %s" % model_type)

        self.eval()
        sd = self.state_dict()

        if _self : 
            checkpoint=torch.load(model_type,map_location=self.config.device)
            sd_hf=checkpoint["model"]

            self.load_state_dict(sd_hf)
            
            return self

           

        else : 
            from transformers import BlipForConditionalGeneration

            # init a huggingface/transformers model
            model_hf =BlipForConditionalGeneration.from_pretrained(model_type)
            sd_hf = model_hf.state_dict()

            sd_keys_hf = [k for k in sd_hf.keys() if not k.startswith('vision_model.')] # ignore vision parameters
            sd_keys_hf = [k for k in sd_keys_hf if not k.startswith('text_decoder.cls.')] # Only care about the last layer
            sd_keys_hf = [k for k in sd_keys_hf if not k.startswith('text_decoder.bert.embeddings.')] # Only care about the last layer
            ## Forced to do this, name doesn't match
            i=0
            for j,k in enumerate(sd_keys_hf) :

                    if i>self.config.config_cross.n_layers -1  :
                        break
            
                    if k.endswith(f'{i}.crossattention.self.query.weight'):
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.co_cross_attn_audio.self.q.weight'].copy_(sd_hf[k])
                    
                    
                    elif k.endswith(f'{i}.crossattention.self.query.bias') :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.co_cross_attn_audio.self.q.bias'].copy_(sd_hf[k])
                    
                            
                    elif k.endswith(f'{i}.crossattention.self.key.bias') :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.co_cross_attn_audio.self.k.bias'].copy_(sd_hf[k])
                                
                    
                    
                    
                    elif k.endswith(f'{i}.crossattention.self.key.weight') :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.co_cross_attn_audio.self.k.weight'].copy_(sd_hf[k])
                                
                    
                    elif k.endswith(f'{i}.crossattention.self.value.weight') :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.co_cross_attn_audio.self.v.weight'].copy_(sd_hf[k])
                                
                    
                    elif k.endswith(f'{i}.crossattention.self.value.bias') :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.co_cross_attn_audio.self.v.bias'].copy_(sd_hf[k])
                                
                        
                    elif k.endswith(f'{i}.crossattention.output.dense.weight') :
                        with torch.no_grad():
                
                                sd[f'cross_module.co_cross.{i}.co_cross_attn_audio.output.dense.weight'].copy_(sd_hf[k])
                                
                    elif k.endswith(f'{i}.crossattention.output.dense.bias') :
                        with torch.no_grad():
                
                                sd[f'cross_module.co_cross.{i}.co_cross_attn_audio.output.dense.bias'].copy_(sd_hf[k])
                                
                    elif k.endswith(f'{i}.crossattention.output.bias') :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.co_cross_attn_audio.output.dense.bias'].copy_(sd_hf[k])
                                
                                
                    elif k.endswith(f'{i}.crossattention.output.LayerNorm.weight')  :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.co_cross_attn_audio.output.LayerNorm.weight'].copy_(sd_hf[k])
                                        
                    elif k.endswith(f'{i}.crossattention.output.LayerNorm.bias')  :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.co_cross_attn_audio.output.LayerNorm.bias'].copy_(sd_hf[k])
                                                    
                                
                    elif k.endswith(f'{i}.attention.self.query.weight') :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.sa_audio.self.q.weight'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.sa_video.self.q.weight'].copy_(sd_hf[k])
                    
                                
                    
                    elif k.endswith(f'{i}.attention.self.key.weight') :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.sa_audio.self.k.weight'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.sa_video.self.k.weight'].copy_(sd_hf[k])
                                
                        
                    elif k.endswith(f'{i}.attention.self.value.weight')     :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.sa_audio.self.v.weight'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.sa_video.self.v.weight'].copy_(sd_hf[k])
                    
                    
                    elif k.endswith(f'{i}.attention.self.query.bias') :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.sa_audio.self.q.bias'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.sa_video.self.q.bias'].copy_(sd_hf[k])
                    
                                
                    
                    elif k.endswith(f'{i}.attention.self.key.bias')  :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.sa_audio.self.k.bias'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.sa_video.self.k.bias'].copy_(sd_hf[k])
                                
                        
                    elif k.endswith(f'{i}.attention.self.value.bias') :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.sa_audio.self.v.bias'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.sa_video.self.v.bias'].copy_(sd_hf[k])
                                    
                    elif k.endswith(f'{i}.attention.output.dense.bias') :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.sa_audio.output.dense.bias'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.sa_video.output.dense.bias'].copy_(sd_hf[k])
                                
                    
                    elif k.endswith(f'{i}.attention.output.dense.weight') :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.sa_audio.output.dense.weight'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.sa_video.output.dense.weight'].copy_(sd_hf[k])
                                
                    
                    elif k.endswith(f'{i}.output.dense.weight' )  :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.audio_output.dense.weight'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.video_output.dense.weight'].copy_(sd_hf[k])
                    
                    
                    elif k.endswith(f'{i}.attention.output.LayerNorm.weight')  :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.sa_audio.output.LayerNorm.weight'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.sa_video.output.LayerNorm.weight'].copy_(sd_hf[k])
                                        
                    elif k.endswith(f'{i}.attention.output.LayerNorm.bias')  :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.sa_audio.output.LayerNorm.bias'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.sa_video.output.LayerNorm.bias'].copy_(sd_hf[k])
                    
                    
                    elif k.endswith(f'{i}.output.LayerNorm.bias')  :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.audio_output.layer_norm.bias'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.video_output.layer_norm.bias'].copy_(sd_hf[k])
                        i+=1            
                
                
                    elif k.endswith(f'{i}.output.LayerNorm.weight')  :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.audio_output.layer_norm.weight'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.video_output.layer_norm.weight'].copy_(sd_hf[k])
                    
                    elif k.endswith(f'{i}.intermediate.dense.weight')  :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.audio_inter.dense.weight'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.video_inter.dense.weight'].copy_(sd_hf[k])
                                            
                                                    
                    elif k.endswith(f'{i}.output.dense.bias')  :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.audio_output.dense.bias'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.video_output.dense.bias'].copy_(sd_hf[k])

                    elif k.endswith(f'{i}.intermediate.dense.weight')  :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.audio_inter.dense.weight'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.video_inter.dense.weight'].copy_(sd_hf[k])
                                
                                
                    elif k.endswith(f'{i}.intermediate.dense.bias')  :
                        with torch.no_grad():
                                sd[f'cross_module.co_cross.{i}.audio_inter.dense.bias'].copy_(sd_hf[k])
                                sd[f'cross_module.co_cross.{i}.video_inter.dense.bias'].copy_(sd_hf[k])
                    
                    else :
                        print("Key Not found",k)
                                            
          

