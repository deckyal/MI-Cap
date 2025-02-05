import torch
from dataclasses import dataclass
from transformers import BlipTextConfig, BlipVisionConfig,BlipConfig, PretrainedConfig
from peft import LoraConfig



@dataclass
class ConfigCross  :
    hidden_size: int =768
    num_heads : int =12
    n_layers : int =2
    hidden_dropout_prob: float = 0.0
    attn_dropout : float = 0.0
    factor_hidden_state : int = 4
    hidden_act :str ="gelu"
    eps : float = 1e-12
  


@dataclass
class ConfigFullModel :


    config_cross=ConfigCross()
   
    lora_config=LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "self.query",
        "self.key",
        "self.value",
        "output.dense",
        "self_attn.qkv",
        "self_attn.projection",
        "mlp.fc1",
        "mlp.fc2",
    ],)


    blip_model='./model_base.pth'
   
    batch_size : int =14
    grad_accum_steps : int= 2
    num_workers : int =0
    init_lr :float = 1e-5

    weigth_decay :float =0.02
    max_epoch : int = 30
    min_lr=1e-6

    lr_decay_rate=0.9
    warmup_lr=1e-6
    
    


    ### Pooler
    hidden_size : int = config_cross.hidden_size
    embed_dim : int =256
    n_frame=5
    

    ## Model initialization
    initializer_range : float = 0.02
    save_grad : bool =False
    gate= False

    ### Model generation 
    k : int =50
    max_length : int =30
    p : float = 0.9
    temperature: float=0.7

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

