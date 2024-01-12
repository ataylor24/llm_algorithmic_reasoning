import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, LEDForConditionalGeneration, LongformerConfig, \
    LongformerTokenizerFast, LongformerModel, LlamaForCausalLM, LlamaTokenizer, BertTokenizer, BertModel

logger = logging.getLogger(__name__)

class GenerativeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model == "llama":
            model = LlamaForCausalLM.from_pretrained(config.llama).to(config.device)
            
            tokenizer = LlamaTokenizer.from_pretrained(config.llama)
    
            self.model_pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device=config.device
                )
        else:
            raise NotImplementedError("Attempting to use a model that hasn't been implemented yet!")
    def forward(self, batch, predict=False):

        outputs = self.model(input_ids=batch.enc_idxs, 
                             attention_mask=batch.enc_attn, 
                             decoder_input_ids=batch.dec_idxs, 
                             decoder_attention_mask=batch.dec_attn, 
                             labels=batch.lbl_idxs, 
                             return_dict=True)
        
        generative_loss = outputs['loss']
        
        if 'ET' in self.config.task_list and 'token_cls' in self.config.tasks['ET']['design']:
            cls_head_in = outputs['encoder_last_hidden_state']
            classifier_logits = self.classification_head(cls_head_in)
            if predict:
               return classifier_logits
            # classifier_logits size: [8, 186, 3]
            # batch.entity_labels size: [8, 186]
            classification_loss = self.cls_loss_fct(
                                classifier_logits.view(-1, self.num_labels), 
                                batch.entity_labels.view(-1))
            if len(self.config.task_list) == 1:
                # only perform seq tagging ET
                loss = classification_loss #* self.cls_loss_weight
            else:
                loss = generative_loss + classification_loss #* self.cls_loss_weight
        else:
            classification_loss = 0
            loss = generative_loss
        # DataParallel cannot handle multiple output
        # return loss, generative_loss, classification_loss
        return loss
