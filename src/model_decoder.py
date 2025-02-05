
import torch
import torch.nn as nn   
from transformers import BertConfig,BertModel,BertLMHeadModel,BertTokenizer


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    return tokenizer



class CustomDecoder(nn.Module):
    def __init__(self,                 
                 med_config = './Modules/config/config.json',  
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
       
        med_config = BertConfig.from_json_file(med_config)
        self.tokenizer=init_tokenizer()
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=med_config)    
        self.text_decoder.resize_token_embeddings(len(self.tokenizer)) 

        


        
    def forward(self, encoder_hidden_states,input_ids,decoder_attention_mask,output_attentions):
        
        image_atts = torch.ones(encoder_hidden_states.size()[:-1],dtype=torch.long).to(encoder_hidden_states.device)
        
        
        input_ids[:,0] = self.tokenizer.bos_token_id
        decoder_targets = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)         
     
        decoder_output = self.text_decoder(input_ids, 
                                           attention_mask = decoder_attention_mask, 
                                           encoder_hidden_states = encoder_hidden_states,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True, 
                                           output_attentions =output_attentions
                                          )   
        
        return decoder_output
        
   