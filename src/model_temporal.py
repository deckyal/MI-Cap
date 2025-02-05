
import torch
import torch.nn as nn 
import torch.nn.functional as F
from typing import Optional,Tuple


class VisionSelfAttention(nn.Module) :
    def __init__(self, config) :
        super().__init__()
        self.embd_dim=config.hidden_size
        self.num_head=config.num_head
        self.head_size=self.embd_dim//self.num_head

        self.scale=self.head_size**0.5

        self.qkv=nn.Linear(self.embd_dim, 3*self.embd_dim)
        self.projection=nn.Linear(self.embd_dim,self.embd_dim)

        self.dropout=nn.Dropout(p=config.attn_dropout)
    

    def forward(self,
            pixel_values : torch.tensor,
            attention_mask : Optional[torch.tensor] = None,
            output_attentions : Optional[bool] =False,
    ) -> Tuple[torch.tensor,Optional[torch.tensor]]:
        
        p,bnf,embd_dim=pixel_values.shape
        mixed_qkv=(
            self.qkv(pixel_values).reshape(p,bnf,3,self.num_head,self.head_size)
            .permute(2,0,3,1,4) ## 3,B,Nheads,P², head_dim
        )

        mixed_q,mixed_k,mixed_v=mixed_qkv[0],mixed_qkv[1],mixed_qkv[2]

        attention_scores=torch.matmul(mixed_q,mixed_k.transpose(-1,-2))

        attention_scores=attention_scores*self.scale
        attention_probs=F.softmax(attention_scores,dim=-1)

        attention_probs=self.dropout(attention_probs)
        if attention_mask is not None: 
            attention_probs=attention_probs +attention_mask

        context_layer=torch.matmul(attention_probs,mixed_v).permute(0,2,1,3)

        new_shape=context_layer.size()[:2] + (embd_dim,)
        context_layer=context_layer.reshape(new_shape)

        output=self.projection(context_layer)
        outputs=(output,)
        if output_attentions :
            outputs=(output,attention_probs)

        return outputs
    

class MLP(nn.Module) :
    def __init__(self, config) :
        super().__init__()

        self.fc1=nn.Linear(config.hidden_size,config.hidden_size * config.factor_hidden_state)
        self.fc2=nn.Linear(config.hidden_size * config.factor_hidden_state,config.hidden_size)
        self.activation_fn=nn.GELU()

    def forward(self, hidden_state: torch.tensor) -> torch.tensor :

        return self.fc2(self.activation_fn(self.fc1(hidden_state)))


class TemporalEncoderLayer(nn.Module) :
    def __init__(self, config) :
        super().__init__()
        self.self_attn=VisionSelfAttention(config)
        self.layer_norm1=nn.LayerNorm(config.hidden_size)
        self.mlp=MLP(config)

        self.layer_norm2=nn.LayerNorm(config.hidden_size)
    
    def forward(self, hidden_state : torch.tensor, 
                attention_mask :Optional[torch.tensor]=None,
                output_attentions :Optional[bool]=False) ->Tuple[torch.tensor,Optional[torch.tensor]]   :

        residual=hidden_state

        hidden_state=self.layer_norm1(hidden_state)

        if output_attentions :        
            hidden_state, attn_probs=self.self_attn(
            hidden_state,
            attention_mask,
            output_attentions
            )
        else :
            hidden_state=self.self_attn(
            hidden_state,
            attention_mask,
            output_attentions
            )[0]

        
        hidden_state=hidden_state+residual

        residual=hidden_state
        hidden_state=self.layer_norm2(hidden_state)
        hidden_state=self.mlp(hidden_state)

        hidden_state=hidden_state+residual
        
        outputs=(hidden_state,)
        if output_attentions  :
            outputs+=(attn_probs,)
        

        return outputs
    



class VisualEmbedding(nn.Module) :
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_embeddings):
        seq_length = input_embeddings.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(input_embeddings.size(0), -1)

        words_embeddings = self.word_embeddings(input_embeddings)


        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings[:,None,:] + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class TemporalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config=config
        self.embding=VisualEmbedding(config)
        self.encoder=nn.ModuleList([TemporalEncoderLayer(config) for _ in range(config.n_layers)])
        self.post_layernorm=nn.LayerNorm(config.hidden_size)


    def forward(self, video_features : torch.tensor,
                attention_mask : Optional[torch.tensor]=None, 
                output_attentions : Optional[bool]=False) -> Tuple[torch.tensor,Optional[torch.tensor]]:        

        B,S=video_features.shape

        video_feat=video_features
        video_features=self.embding(video_feat)

        for encoder in self.encoder :
            video_features,*other=encoder(video_features,attention_mask,output_attentions)

        

        outputs=(self.post_layernorm(video_features).view(B,-1,S),)
        if output_attentions : 
            outputs+=other

        return outputs