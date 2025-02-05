import torch
import torch.nn as nn
import torch.nn.functional as F






class AggModule(nn.Module) :
    def __init__(self, config) :
        super().__init__()

        self.config=config
        if self.config.gate :
            self.mlp=nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.hidden_size,eps=1e-12),
                
            )
        self.post_ln=nn.LayerNorm(config.hidden_size)
        
    
      
    def forward(self, 
                hidden_state_vid : torch.tensor,
                hidden_state_audio: torch.tensor,
                ) :
        
      
            
        
        hidden_state=torch.cat([hidden_state_audio,hidden_state_vid],dim=1)
        if self.config.gate :
            return self.mlp(hidden_state)
        
        hidden_state=self.post_ln(hidden_state)



        return hidden_state




