import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional,Tuple
from transformers import PreTrainedModel

from Modules.util_config import ConfigCross
import math


ACT={"gelu" :nn.GELU(), "relu" :nn.ReLU()}



##########################################################################################

def get_sinusoidal_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    pos_encoding = torch.zeros(seq_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    
    return pos_encoding


class CrossPosEmeddings(nn.Module) :
    def __init__(self, d_model, max_seq_len=800):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.encodings = get_sinusoidal_encoding(max_seq_len, d_model)

    def forward(self, x, start_index=0):
        seq_len = x.size(1)
        return self.encodings[start_index:start_index+seq_len, :].to(x.device)
    
6


class CrossEmbeddings(nn.Module):
    def __init__(self, config_cross) :
        super().__init__()
        
        self.position_embeddings = nn.Embedding(1024,config_cross.hidden_size)
        self.token_embeddings =nn.Embedding(config_cross.hidden_size,config_cross.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config_cross.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config_cross.hidden_dropout_prob)

    def forward(self, concat_embeddings,concat_type=None):

        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)
        if concat_type is None :
            concat_type=torch.zeros((batch_size,seq_length),dtype=torch.long).to(concat_embeddings.device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(concat_embeddings.size(0), -1)
       
        token_type_embeddings = self.token_embeddings(concat_type)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = concat_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings






class CoModalSelfAttention(nn.Module) :
    def __init__(self, config_cross) :
        super().__init__()

        self.num_heads = config_cross.num_heads
        self.head_size = config_cross.hidden_size // config_cross.num_heads

        self.scale=self.head_size**-0.5
        self.q= nn.Linear(config_cross.hidden_size, config_cross.hidden_size)
        self.k=nn.Linear(config_cross.hidden_size, config_cross.hidden_size)
        self.v=nn.Linear(config_cross.hidden_size, config_cross.hidden_size)

        # self.rope_q=RoPe(self.dq)
        # self.rope_k=RoPe(self.dkv)

      
        self.dropout = nn.Dropout(config_cross.attn_dropout)
    

    def forward(self,
                hidden_state: torch.tensor, 
                context: torch.tensor,
                attention_mask : Optional[torch.tensor]=None,
                output_attentions : Optional[bool]=None
                ) -> Tuple[torch.tensor,Optional[torch.tensor]]:
        
        B,N,D=hidden_state.shape
        B,L,D=context.shape
       
        mixed_q=self.q(hidden_state).view(B,N,self.num_heads,self.head_size).permute(0,2,1,3)
        mixed_k, mixed_v=self.k(context).view(B,L,self.num_heads,self.head_size).permute(0,2,1,3) ,self.v(context).view(B,L,self.num_heads,self.head_size).permute(0,2,1,3)


        # xq=self.rope_q(q)
        # xk=self.rope_k(k)

        attention_scores=torch.matmul(mixed_q,mixed_k.transpose(-1,-2))

        attention_scores=attention_scores*self.scale
        if attention_mask is not None: 
            attention_scores=attention_scores+attention_mask.to(attention_scores.device)

        attention_probs=F.softmax(attention_scores,dim=-1)
        attention_probs=self.dropout(attention_probs)

        context_layer=torch.matmul(attention_probs,mixed_v).permute(0,2,1,3)

        new_shape=context_layer.size()[:2] + (D,)
        context_layer=context_layer.reshape(new_shape)

        output=(context_layer,attention_probs) if output_attentions else (context_layer,None)

        return output


class CoModalAttOutput(nn.Module) :
    def __init__(self, config_cross):
        super().__init__()
        self.dense = nn.Linear(config_cross.hidden_size, config_cross.hidden_size)
        self.dropout = nn.Dropout(config_cross.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config_cross.hidden_size, eps=1e-12)

    def forward(self, 
                hidden_states : torch.tensor, 
                input_tensor: torch.tensor)->torch.tensor:
                
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class CoModalSelfAttLayer(nn.Module) : 
    def __init__(self, config_cross) : 
        super().__init__()
        self.self=CoModalSelfAttention(config_cross)
        self.output=CoModalAttOutput(config_cross)
    
    def forward(self,
                input_tensor : torch.tensor,
                attention_mask : Optional[torch.tensor] =None,
                output_attentions : Optional[bool] = True) ->Tuple[torch.tensor, Optional[torch.tensor]]:
        
        output_ca, attn_weights=self.self(input_tensor,
                                    context=input_tensor,
                                    attention_mask=attention_mask,
                                    output_attentions=output_attentions)
        
        attn_output=self.output(output_ca, input_tensor)

        return attn_output,attn_weights


class CoModalCrossAttLayer(nn.Module) :
     def __init__(self, config_cross) : 
        super().__init__()
        self.self=CoModalSelfAttention(config_cross)
        self.output=CoModalAttOutput(config_cross)
                 
    
     def forward(self,
                input_tensor : torch.tensor,
                context_tensor : torch.tensor,
                attention_mask :Optional[torch.tensor] = None,
                output_attentions : Optional[bool] = True) ->Tuple[torch.tensor,Optional[torch.tensor]]:
        
        output_ca_video, ca_weights=self.self(hidden_state=input_tensor,
                                    context=context_tensor,
                                    attention_mask=attention_mask,
                                    output_attentions=output_attentions)
        
        attn_output_video=self.output(output_ca_video, input_tensor)

        outupts=(attn_output_video,ca_weights) if output_attentions else (attn_output_video,None)
  


        return outupts


class CoModalIntermediate(nn.Module) :
    def __init__(self, config_cross ) :
        super().__init__()
        self.config=config_cross
        self.dense=nn.Linear(config_cross.hidden_size, config_cross.hidden_size * config_cross.factor_hidden_state)
        self.intermediate_act_fn=ACT[config_cross.hidden_act]
    

    def forward(self, 
                hidden_state: torch.tensor) ->torch.tensor:
        
        hidden_state=self.intermediate_act_fn(self.dense(hidden_state))
    
        return hidden_state
    

class CoModalOutput(nn.Module):
    def __init__(self, config_cross) :
        super().__init__()

        self.dense=nn.Linear(config_cross.hidden_size * config_cross.factor_hidden_state, config_cross.hidden_size)
        self.droptoken=nn.Dropout(config_cross.hidden_dropout_prob)
        self.layer_norm=nn.LayerNorm(config_cross.hidden_size, config_cross.eps)
    
    def forward(self, 
                hidden_state: torch.tensor,
                input_tensor : torch.tensor) ->torch.tensor: 
        
        hidden_state=self.dense(hidden_state)
        hidden_state=self.droptoken(hidden_state)
        hidden_state=self.layer_norm(hidden_state + input_tensor)

        return hidden_state

        

class CoModalCrossBlock(nn.Module):
    def __init__(self, config_cross ) :
        super().__init__()     

        self.config=config_cross
        

        self.co_cross_attn_audio=CoModalCrossAttLayer(config_cross)

        self.sa_video=CoModalSelfAttLayer(config_cross)
        self.video_inter=CoModalIntermediate(config_cross)
        self.video_output=CoModalOutput(config_cross)
        
        self.sa_audio=CoModalSelfAttLayer(config_cross)
        self.audio_inter=CoModalIntermediate(config_cross)
        self.audio_output=CoModalOutput(config_cross)


    
    def _cross_attentions(self,input_tensor,context_tensor,attention_mask,output_attentions) :
        
         context, cross_attn=self.co_cross_attn_audio(input_tensor=input_tensor,
                                                                context_tensor=context_tensor,
                                                                attention_mask=attention_mask,
                                                                output_attentions=output_attentions)
         
         
         return (context,cross_attn) if output_attentions else (context,None)
    
       
    def _self_attentions(self,is_audio,input_tensor,attention_mask,output_attentions) :
        
        if is_audio :
                context, self_attn_probs=self.sa_audio(input_tensor=input_tensor,attention_mask=attention_mask, output_attentions=output_attentions,)
        else :
            
                context, self_attn_probs=self.sa_audio(input_tensor=input_tensor,attention_mask=attention_mask, output_attentions=output_attentions,)

            
        return (context,self_attn_probs) if output_attentions else (context,None)
    
    def forward(self,video_features : torch.tensor,
                audio_features : torch.tensor,
                video_attention_mask : Optional[torch.tensor] = None,
                audio_attention_mask : Optional[torch.tensor]= None,
                output_attentions : Optional[bool]=False) ->Tuple[torch.tensor,
                                                                 torch.tensor,Optional[torch.tensor],
                                                                 Optional[torch.tensor]
                                                                 ,Optional[torch.tensor],Optional[torch.tensor]]:
        
        

       
        context_video, cross_attn_video_audio=self._cross_attentions(input_tensor=video_features,
                                                                context_tensor=audio_features,
                                                                attention_mask=audio_attention_mask,
                                                                output_attentions=output_attentions)
    
        context_audio, cross_attn_audio_video=self._cross_attentions(input_tensor=audio_features,
                                                                context_tensor=video_features,
                                                                attention_mask=video_attention_mask,
                                                                output_attentions=output_attentions)


                                                        
    

        context_audio, self_attn_probs_audio=self._self_attentions(is_audio=True,input_tensor=context_audio,attention_mask=audio_attention_mask, output_attentions=output_attentions)

       
        context_video, self_attn_probs_video= self._self_attentions(is_audio=False,input_tensor=context_video,attention_mask=video_attention_mask,output_attentions=output_attentions)
        
        residual_video=context_video
        residual_audio=context_audio
       
        context_video=self.video_inter(context_video)
        context_video=self.video_output(context_video,residual_video)


        context_audio=self.audio_inter(context_audio)
        context_audio=self.audio_output(context_audio,residual_audio)

        outputs=(context_video, context_audio)
        if  output_attentions : 
            cross_attn_tuple=(cross_attn_video_audio,cross_attn_audio_video) 
            outputs +=cross_attn_tuple +(self_attn_probs_audio,self_attn_probs_video) if self_attn_probs_video is not None else cross_attn_tuple +(self_attn_probs_audio,None)
        
            
     
        return outputs
        
class CrossPooler(nn.Module):
    def __init__(self, config) :
        super().__init__()

        self.config=config
       
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, 
                hidden_state: torch.tensor,
                ) :
        
    
        pooled=self.activation(self.dense(hidden_state[:,0,:]))

       

        return pooled
    


class CoModalModule(nn.Module) :
    
    def __init__(self, config_cross) -> None:
        super().__init__()

        self.audio_emebdding=CrossEmbeddings(config_cross)
        self.video_emebdding=CrossEmbeddings(config_cross)
        self.co_cross=nn.ModuleList([CoModalCrossBlock(config_cross) for _ in range(config_cross.n_layers)])
        self.cross_pooler_audio=CrossPooler(config_cross)
        self.cross_pooler_video=CrossPooler(config_cross)



    def forward(self,
                video_feat: torch.tensor,
                audio_feat : torch.tensor,
                video_attention_mask: Optional[torch.tensor]=None,
                audio_attention_mask : Optional[torch.tensor]= None,
                output_attentions : Optional[bool]=False
                ) :
        
    
        video_feat=self.video_emebdding(video_feat,concat_type=torch.ones(video_feat.shape[:-1],dtype=torch.long).to(video_feat.device))
        audio_feat=self.audio_emebdding(audio_feat)

      
        if audio_attention_mask is not None :
            audio_attention_mask=(1.0-audio_attention_mask[:,None,None,:])*-10000.0

        if video_attention_mask is  None : 
            B,N,D=video_feat.shape

            video_attention_mask=torch.ones(B,N,dtype=torch.long)
            video_attention_mask=video_attention_mask[:,None,None,:]  

        video_attention_mask = (1.0 - video_attention_mask) * -10000.0

        cross_attn_tuple_all=() if output_attentions else None
        self_attn_tuple_all=() if output_attentions else None

        for layer in self.co_cross :
            video_feat,audio_feat,*other=layer(video_features=video_feat,
                         audio_features=audio_feat,
                         video_attention_mask=video_attention_mask,
                         audio_attention_mask=audio_attention_mask,
                        output_attentions=output_attentions)



        
        video_features, audio_features= video_feat,audio_feat
        if output_attentions :
            cross_attn_tuple_all+=tuple(other)[:2]
            self_attn_tuple_all+=tuple(other)[2:]

        
        pooled_audio=self.cross_pooler_audio(audio_features)
        pooled_video=self.cross_pooler_video(video_feat)

        return video_features,audio_features,pooled_video,pooled_audio,cross_attn_tuple_all,self_attn_tuple_all

            
        