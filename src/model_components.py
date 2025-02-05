import torch
import torch.nn as nn 
import torch.nn.functional as F
from transformers import BlipForConditionalGeneration,CLIPVisionModel
from transformers import BertModel,BertConfig,BertForMaskedLM
from typing import Optional,Tuple
import numpy as np
from peft import get_peft_model,PeftConfig,PeftModel,LoraConfig
from transformers import AutoConfig
import timm
from .get_blip import load_blip
from .CLIP.clip.model import VisionTransformer,LayerNorm


class IFrameEncoder(VisionTransformer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                ):
        super().__init__(
            input_resolution=input_resolution, patch_size=patch_size, width=width, layers=layers, heads=heads,
            output_dim=output_dim
        )

        scale = width ** -0.5
        self.ln_post_hidden = LayerNorm(width)
        self.proj_hidden = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, output_all_features: bool = True):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid = x.size(2)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        
        outputs = self.ln_post_hidden(x) @ self.proj_hidden

        return outputs

    
    @classmethod
    def from_pretrained(cls,model_path):
        
        clip_model=torch.load(model_path)
        
        state_dict = clip_model.state_dict()

        vision_width: int = state_dict["visual.conv1.weight"].shape[0]
        
        vision_layers: int = len([k for k in state_dict.keys()
                                  if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        
        
        embed_dim: int = state_dict["text_projection"].shape[1]
        vision_patch_size: int = state_dict["visual.conv1.weight"].shape[-1]
        grid_size: int = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution: int = vision_patch_size * grid_size
        vision_heads = vision_width // 64
        
        frame_encoder=cls(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width, layers=vision_layers, heads=vision_heads,
            output_dim=embed_dim if embed_dim==1024 else 768
        )
        
        visual_state_dict = frame_encoder.state_dict()
        # manually build the state dict to make sure pretrained weights are loaded exactly
        visual_state_dict.update({k: v for k, v in frame_encoder.state_dict().items() if k.startswith("ln_post_hidden")})
        visual_state_dict.update({k: v for k, v in frame_encoder.state_dict().items() if k.startswith("proj_hidden")})
        frame_encoder.load_state_dict(visual_state_dict, strict=True)

        return frame_encoder
        

 
class VideoEncoder(nn.Module):
    def __init__(self, blip,blip_model=None,n_frame=4):
        super().__init__()

        if blip : 
            
            self.vision_model=load_blip(blip_model,decoder=False).visual_encoder
            
            
        
        else :
           self.vision_model=IFrameEncoder.from_pretrained('./ViT-L-14.pt')

            
        
        self.img_temperal_embedding = nn.ParameterList(
                nn.Parameter(torch.zeros(1, 1, 768))
                for _ in range(n_frame)
            )
        
        self.visual_projection = nn.Sequential(
            nn.Linear(768,768),
            nn.LayerNorm(768,1e-12),
        )
        

    
    def forward(self, 
                frames : torch.tensor,
              ) :


        B=frames.shape[0]
        visual_features=list()
        for frame_idx in range(frames.shape[1]) :
            
            visual_features_frame = self.vision_model(frames[:, frame_idx, :, :]) 
            visual_features_frame += self.img_temperal_embedding[frame_idx]
            visual_features.append(visual_features_frame)
            
            
        visual_features = torch.cat(visual_features, dim=1)
        visual_features=self.visual_projection(visual_features)


        return visual_features




class AudioCaptionModel(nn.Module):
    def __init__(self, bert) -> None:
        super().__init__()
        """
            Blip text encoder with bi-self attention 
        """

        
        if bert : 
            self.audio_text_model=BertModel.from_pretrained('bert-base-uncased')
        
        else :
       
          
            self.audio_text_model=load_blip("./model_large.pth",decoder=False,vit='large').text_encoder
        
    
           
           
    def forward(self, audio_caption_ids : torch.LongTensor, attention_mask : torch.LongTensor,
              ) :

    
        out= self.audio_text_model(input_ids=audio_caption_ids,
                                    attention_mask= attention_mask,   
                                    mode='text'
                                     )
        
        
        return out
    


class AudioLMHead(nn.Module) :
    def __init__(self, embedding_weights):
        super().__init__()
        self.hidden_size = embedding_weights.size(1)
        self.vocab_size = embedding_weights.size(0)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size)
        self.decoder.weight = embedding_weights
        


    def forward(self, sequence_output):
    
        sequence_output = self.dense(sequence_output)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.decoder(sequence_output)

        return prediction_scores


class BlipDecoder(nn.Module):
    def __init__(self, name,config_lora:Optional[LoraConfig]=None) :
        super().__init__()

       
        if config_lora is not None : 
            self.decoder=get_peft_model(BlipForConditionalGeneration.from_pretrained(name),config_lora).text_decoder
        else : 
            from transformers import BlipConfig

            config = AutoConfig.from_pretrained(name)
            config.text_config.num_hidden_layers=3
            self.decoder=BlipForConditionalGeneration(config).text_decoder
           

    

    

    def forward(self,input_ids: torch.LongTensor, 
                attention_mask : torch.LongTensor, 
                encoder_hidden_state : torch.tensor, 
                encoder_attention_mask: Optional[torch.tensor]=None,
                output_attentions : Optional[bool]=False,
                labels  : Optional[torch.tensor] =None) ->Tuple[torch.tensor,Optional[torch.tensor]] :
        
        
        image_embds=encoder_hidden_state
        B,L_,D=image_embds.shape
        
        if encoder_attention_mask is None  :
            encoder_attention_mask=torch.ones(B,L_)
            encoder_extended_attention_mask=encoder_attention_mask[:,None,None,:]
            encoder_extended_attention_mask = encoder_extended_attention_mask.to(device=image_embds.device)  
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(image_embds.dtype).min

        outputs=self.decoder(input_ids=input_ids,
                                attention_mask=attention_mask,
                               encoder_hidden_states=image_embds,
                                encoder_attention_mask=encoder_attention_mask,
                                labels=labels,
                                output_attentions=output_attentions)


        return outputs


 
from transformers import BertLMHeadModel

class SmallDecoder(BertLMHeadModel):
    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        output_dict ={"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "past_key_values": past,
                      "encoder_hidden_states": model_kwargs['encoder_hidden_states']}

        return output_dict

    

    

        
        








        





        







