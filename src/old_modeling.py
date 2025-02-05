

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional,Tuple
from transformers import BertConfig,BertModel,AutoTokenizer

from .model_cross import CoModalModule
from .model_temporal import TemporalEncoder
from .model_components import FrameEncoder,SmallDecoder
from .model_pooler import AggModule


def mlm_loss(logits,labels) :
    
    return F.cross_entropy(logits.view(-1,logits.size(-1)),labels.view(-1),ignore_index=0)



def contrast_loss(cls_video,cls_audio,logits_scale) : 

        video_embd=cls_video/cls_video.norm(p=2,dim=-1,keepdim=True)
        text_embd=cls_audio/cls_audio.norm(p=2,dim=-1,keepdim=True)


        logits_scale=logits_scale.exp()
        logits_per_video=logits_scale*video_embd@text_embd.t()
        logits_per_text=logits_per_video.t()
        

        loss_vt=F.cross_entropy(logits_per_video,torch.arange(len(logits_per_video),device=logits_per_video.device))
        loss_tv=F.cross_entropy(logits_per_text,torch.arange(len(logits_per_text),device=logits_per_text.device))

        loss=(loss_vt+loss_tv)*0.5

        return loss

def init_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer


class FullModelVision(nn.Module) :
    def __init__(self,config_model,stage_two : Optional[bool]=False) :
        super().__init__()

        self.config=config_model


        self.tokenizer=init_tokenizer()
        self.video_encoder=FrameEncoder(config_model.name,n_frame=config_model.n_frame)
        self.video_norm=nn.LayerNorm(config.hidden_size,eps=1e-12)
       



        self.agg_module=AggModule(config_model) 
        config = BertConfig.from_pretrained("bert-base-uncased", bos_token_id=101, pad_token_id=0, eos_token_id=102)
        config.is_decoder = True
        config.add_cross_attention = True
        config.num_hidden_layers=3
        self.text_decoder = SmallDecoder.from_pretrained('bert-base-uncased', config=config)
        self.text_decoder=self._init_weights(self.text_decoder)

        
        self.grad=None
        self.decoder_input_ids = 101
        self.eos_token_ids = 102


                        
    def _init_weights(self,model) :   
        t=torch.load('weight/univl.pretrained.bin')
        for n,p in model.named_parameters() :
            if 'cls' in n :
                n='decoder.classifier.'+n 
            else :
                n='decoder'+n[4:]
            
            if 'encoder' in n :
                n='decoder.decoder'+n[15:]
            
            if n in t.keys() :
                p.data=t[n]
        
        return model
        
                                   
                                      
    @torch.no_grad()
    def prepare_for_generation(self,pixel_values,audio_caption_ids,attention_mask) :
        last_hidden_state_video=self.video_encoder(pixel_values)
     

        return last_hidden_state_video
    
    
    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        
        return video_out,text_out

    def forward(self,
                pixel_values : torch.tensor, 
                audio_caption_ids : torch.LongTensor,
                attention_mask : Optional[torch.LongTensor]=None,
                decoder_input_ids : Optional[torch.tensor]=None,
                decoder_attention_mask: Optional[torch.LongTensor]=None,
                labels : Optional[torch.LongTensor]=None ,
               targets : Optional[torch.tensor] = None,
                output_attentions : Optional[bool]=True,
                alpha : int =1.0
                ) :
       
       

        video_hidden_state=self.video_encoder(pixel_values)
        
        video_hidden_state=self.video_norm(video_hidden_state)
        
        out=self.text_decoder(input_ids=decoder_input_ids,
                            attention_mask=decoder_attention_mask,
                            encoder_hidden_states=video_hidden_state,
                            output_attentions=output_attentions,
                            labels=labels,
                            )

       
        if labels is not None :
          
            loss_model=out.loss
            
        outputs=(loss_model,out.logits)
        

        return  outputs
    
    @torch.no_grad()
    def generate(self, image,audio_caption,audio_attention_mask,input_ids=None, sample=False,topk=False, num_beams=5, max_length=20, min_length=1, top_p=0.9, repetition_penalty=1.0):
       

        image_embeds=self.prepare_for_generation(pixel_values=image,
                                                 audio_caption_ids=audio_caption,
                                                 attention_mask=audio_attention_mask)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts,"output_attentions" : True, "return_dict_in_generate" : True}
        
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[self.decoder_input_ids, self.eos_token_ids]])
                .repeat(image_embeds.size(0), 1)
                .to(image_embeds.device)
            )

        input_ids[:,0] = self.decoder_input_ids
        input_ids = input_ids[:, :-1] 

        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            

                                                  **model_kwargs)
        
        elif topk : 
            print('TOP K')
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  top_k=5,
                                                  
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)       
            
        
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)       
            
       
            
            
        return outputs