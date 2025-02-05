import torch
import torch.nn as nn 
import torch.nn.functional as F
from transformers import PreTrainedModel

from util_config import ConfigModelPretrain


class MultiModalPreTrainedModel(PreTrainedModel) :

    config_class=ConfigModelPretrain



    def init_weights(self, module):
        """ Initialize the weights for the temporal encoder
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    

    def from_pretrained(self,cls, config, state_dict=None, *inputs, **kwargs) :
          # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)

        return model
         