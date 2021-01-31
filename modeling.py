import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_bert import BertModel, BertPreTrainedModel
from transformers.modeling_roberta import RobertaModel
from transformers.configuration_roberta import RobertaConfig
import numpy

class MMLLoss(nn.Module):
    def __init__(self):
        super(MMLLoss, self).__init__()

    def forward(self, input, target, answer_mask):
        prob = F.softmax(input)
        target = torch.sum(F.one_hot(target, 4) * answer_mask.unsqueeze(2), dim=1)
        mml = -torch.log(torch.sum(prob * target, dim=1))
        loss = torch.mean(mml)
        return loss


class MaximizeLoss(nn.Module):
    def __init__(self):
        super(MaximizeLoss, self).__init__()
    
    def forward(self, input, target, answer_mask):
        prob = F.softmax(input)
        target = torch.sum(F.one_hot(target, 4) * answer_mask.unsqueeze(2), dim=1)
        max_prob = torch.max(prob * target, dim=1).values
        loss = torch.mean(-torch.log(max_prob))
        return loss


class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        self.loss_type = config.loss_type
        self.tau = config.tau
        if self.loss_type=='hard-em':
            assert self.tau is not None

        self.init_weights()
        
        self.celoss_fct = nn.CrossEntropyLoss()
        self.mmlloss_fct = MMLLoss()
        self.maxloss_fct = MaximizeLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        answer_mask=None,
        global_step=None,
        rand_num=None,
    ):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here
        
        if labels is None:
            return ouptuts
        
        if answer_mask is None:
            loss = self.celoss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
            return outputs

        if self.loss_type=='highest-only':
            loss = self.celoss_fct(reshaped_logits, labels[:, 0])
            outputs = (loss,) + outputs

        elif self.loss_type=='mml':
            loss = self.mmlloss_fct(reshaped_logits, labels, answer_mask)
        elif self.loss_type=='hard-em':
            if rand_num < min(global_step / self.tau, 0.8):
                loss = self.maxloss_fct(reshaped_logits, labels, answer_mask)
            else:
                loss = self.mmlloss_fct(reshaped_logits, labels, answer_mask)
        else:
            raise NotImplementedError()
        
        outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)




ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}

class RobertaForMultipleChoice(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.loss_type = config.loss_type
        self.tau = config.tau
        if self.loss_type=='hard-em':
            assert self.tau is not None

        self.init_weights()
        
        self.celoss_fct = nn.CrossEntropyLoss()
        self.mmlloss_fct = MMLLoss()
        self.maxloss_fct = MaximizeLoss()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        answer_mask=None,
        global_step=None,
        rand_num=None,
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is None:
            return ouptuts
        
        if answer_mask is None:
            loss = self.celoss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
            return outputs

        if self.loss_type=='highest-only':
            loss = self.celoss_fct(reshaped_logits, labels[:, 0])
            outputs = (loss,) + outputs

        elif self.loss_type=='mml':
            loss = self.mmlloss_fct(reshaped_logits, labels, answer_mask)
        elif self.loss_type=='hard-em':
            if rand_num < min(global_step / self.tau, 0.8):
                loss = self.maxloss_fct(reshaped_logits, labels, answer_mask)
            else:
                loss = self.mmlloss_fct(reshaped_logits, labels, answer_mask)
        else:
            raise NotImplementedError()
        
        outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

