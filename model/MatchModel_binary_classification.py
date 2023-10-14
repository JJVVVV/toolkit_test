import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from toolkit.training import kl_loss
from torch import Tensor
from transformers import BertModel, BertPreTrainedModel, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .mod import WAIO
from .tricks import generate_distribution, generate_distribution2, generate_distribution3, rotate_embeddings, shift_embeddings

# from .mod2 import AttnNoProjVal


# Roberta
#################################################################################################
class RobertaModel_binary_classify(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train=True,
    ) -> dict[str, Tensor]:
        ret = dict()
        outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss
        return ret


class RobertaModel_binary_classify_noise(RobertaModel):
    def __init__(self, config, min_threshold, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.min_threshold = min_threshold

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, min_threshold=None, *model_args, **kwargs):
        kwargs["min_threshold"] = min_threshold
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss

        if is_train:
            dis = generate_distribution2(input_ids=input_ids, vocab_size=self.vocab_size, min_main_score=self.min_threshold)
            # emb = self.roberta.get_input_embeddings().weight.clone()
            emb = self.get_input_embeddings().weight
            input_emb = torch.matmul(dis, emb)
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_emb, output_attentions=False, output_hidden_states=False
            )
            cls2 = outputs2.last_hidden_state[:, 0]
            logits2 = self.classifier(self.tanh(self.pooler(cls2)))
            loss2 = self.loss_func(logits2, labels.float())
            ret["loss"] += loss2
            ret["loss"] /= 2

        return ret


class RobertaModel_binary_classify_shift(RobertaModel):
    def __init__(self, config, alpha, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.alpha = alpha
        self.get_input_embeddings().weight.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, alpha, *model_args, **kwargs):
        kwargs["alpha"] = alpha
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss

        if is_train:
            input_embs = self.get_input_embeddings()(input_ids)
            input_embs = shift_embeddings(input_embs, self.alpha)
            # print(f"#######################\n{input_embs.dtype}")
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_embs, output_attentions=False, output_hidden_states=False
            )
            cls2 = outputs2.last_hidden_state[:, 0]
            logits2 = self.classifier(self.tanh(self.pooler(cls2)))
            loss2 = self.loss_func(logits2, labels.float())
            ret["loss"] += loss2
            ret["loss"] /= 2

        return ret


class RobertaModel_binary_classify_shift_only(RobertaModel):
    def __init__(self, config, alpha, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.alpha = alpha

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, alpha, *model_args, **kwargs):
        kwargs["alpha"] = alpha
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        if is_train:
            input_embs = self.get_input_embeddings()(input_ids)
            input_embs = shift_embeddings(input_embs, self.alpha)
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_embs, output_attentions=False, output_hidden_states=False
            )
        else:
            outputs2 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls2 = outputs2.last_hidden_state[:, 0]
        logits2 = self.classifier(self.tanh(self.pooler(cls2)))
        loss2 = self.loss_func(logits2, labels.float())
        ret["logits"] = logits2
        ret["loss"] = loss2
        return ret


class RobertaModel_binary_classify_rotate(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss

        if is_train:
            input_embs = self.get_input_embeddings()(input_ids)
            input_embs = rotate_embeddings(input_embs)
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_embs, output_attentions=False, output_hidden_states=False
            )
            cls2 = outputs2.last_hidden_state[:, 0]
            logits2 = self.classifier(self.tanh(self.pooler(cls2)))
            loss2 = self.loss_func(logits2, labels.float())
            ret["loss"] += loss2
            ret["loss"] /= 2

        return ret


class RobertaModel_binary_classify_rotate_only(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        if is_train:
            input_embs = self.get_input_embeddings()(input_ids)
            input_embs = rotate_embeddings(input_embs)
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_embs, output_attentions=False, output_hidden_states=False
            )
        else:
            outputs2 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls2 = outputs2.last_hidden_state[:, 0]
        logits2 = self.classifier(self.tanh(self.pooler(cls2)))
        loss2 = self.loss_func(logits2, labels.float())
        ret["logits"] = logits2
        ret["loss"] = loss2
        return ret


class RobertaModel_rephrase(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self, input_ids: Tensor | None = None, attention_mask: Tensor | None = None, position_ids: Tensor | None = None, labels: Tensor = None
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        for i in range(times):
            output = super().forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                position_ids=position_ids,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls = output.last_hidden_state[:, 0]
            logits = self.classifier(self.tanh(self.pooler(cls)))
            logitss.append(logits)
            if labels is not None:
                loss += self.loss_func(logits, labels.float())
        ret["logits"] = torch.stack(logitss, dim=-1).squeeze()
        if labels is not None:
            ret["loss"] = loss
        return ret

    # def forward(
    #     self, input_ids: Tensor | None = None, attention_mask: Tensor | None = None, position_ids: Tensor | None = None, labels: Tensor = None
    # ) -> dict[str, Tensor]:
    #     ret = dict()
    #     times = input_ids.shape[1]
    #     # input_ids: (batch_size, 4, seqence_len)
    #     loss = 0
    #     logitss = []
    #     for i in range(times):
    #         output = super().forward(
    #             input_ids=input_ids[:, i],
    #             attention_mask=attention_mask[:, i],
    #             position_ids=position_ids,
    #             output_attentions=False,
    #             output_hidden_states=False,
    #         )
    #         cls = output.last_hidden_state[:, 0]
    #         logits = self.classifier(self.tanh(self.pooler(cls)))
    #         logitss.append(logits)
    #         # if labels is not None:
    #         #     loss += self.loss_func(logits, labels.float())
    #         if labels is not None:
    #             part_loss = self.loss_func(logits, labels.float()) / times
    #             if i < times - 1:
    #                 part_loss.backward()
    #     ret["logits"] = torch.stack(logitss, dim=1).squeeze()
    #     if labels is not None:
    #         ret["loss"] = part_loss

    #     return ret


class RobertaModel_rephrase_fused(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self, input_ids: Tensor | None = None, attention_mask: Tensor | None = None, position_ids: Tensor | None = None, labels: Tensor = None
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        cls = 0
        for i in range(times):
            output = super().forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                position_ids=position_ids,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls += output.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits
        if labels is not None:
            loss = self.loss_func(logits, labels.float())
            ret["loss"] = loss

        return ret


class RobertaModel_rephrase_close(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, input_ids: Tensor | None = None, attention_mask: Tensor | None = None, position_ids: Tensor | None = None, labels: Tensor = None
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        clss = []
        for i in range(times):
            output = super().forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                position_ids=position_ids,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls = output.last_hidden_state[:, 0]
            clss.append(cls)
            logits = self.classifier(self.tanh(self.pooler(cls)))
            logitss.append(logits)
            if labels is not None:
                loss += self.loss_fn(logits, labels.float())
        for i in range(len(clss) - 1):
            loss += kl_loss(clss[i], clss[i + 1], 1)
        ret["logits"] = torch.stack(logitss, dim=1).squeeze()
        ret["loss"] = loss

        return ret


# class RobertaModel_6times_bi(RobertaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = RobertaModel(config)
#         # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss()

#     def forward(self, input_ids, attention_mask, labels=None, is_training=True):
#         ret = dict()
#         times = input_ids.shape[1]
#         # input_ids: (batch_size, 4, seqence_len)
#         loss = 0
#         logitss = []
#         for i in range(times):
#             output = self.roberta(input_ids=input_ids[:, i], attention_mask=attention_mask[:, i], output_attentions=False)
#             cls = output.last_hidden_state[:, 0]
#             logits = self.classifier(self.tanh(self.pooler(cls)))
#             logitss.append(logits)
#             if labels is not None:
#                 loss += self.loss_func(logitss[i], labels.float()) if i < 4 else self.loss_func(logitss[i], torch.ones_like(labels).float())

#         ret["logits"] = torch.stack(logitss[:4], dim=1).squeeze()
#         ret["loss"] = loss

#         return ret


# class RobertaModel_4times_4classifier_bi(RobertaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = RobertaModel(config)
#         # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.poolers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(4)])
#         self.tanh = nn.Tanh()
#         self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(4)])
#         self.loss_func = nn.BCEWithLogitsLoss()

#     def forward(self, input_ids, attention_mask, labels=None, **kwargs):
#         ret = dict()
#         # input_ids: (batch_size, 4, seqence_len)
#         outputs = []
#         for i in range(4):
#             outputs.append(self.roberta(input_ids=input_ids[:, i], attention_mask=attention_mask[:, i], output_attentions=False))
#         clss = [output.last_hidden_state[:, 0] for output in outputs]

#         logitss = [self.classifiers[i](self.tanh(self.poolers[i](cls))) for cls in clss]
#         ret["logits"] = logitss

#         if labels is None:
#             return logitss

#         losses = [self.loss_func(logits, labels.float()) for logits in logitss]
#         ret["loss"] = sum(losses)
#         return ret

# bert
########################################################################################################################################################


class BertModel_binary_classify(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs.last_hidden_state[:, 0]
        # ret["cls_hidden_state"] = cls
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss
        return ret


class BertModel_binary_classify_noise(BertModel):
    def __init__(self, config, min_threshold, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.min_threshold = min_threshold

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, min_threshold=None, *model_args, **kwargs):
        kwargs["min_threshold"] = min_threshold
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss

        if is_train:
            dis = generate_distribution2(input_ids=input_ids, vocab_size=self.vocab_size, min_main_score=self.min_threshold)
            # emb = self.roberta.get_input_embeddings().weight.clone()
            emb = self.get_input_embeddings().weight
            input_emb = torch.matmul(dis, emb)
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_emb, output_attentions=False, output_hidden_states=False
            )
            cls2 = outputs2.last_hidden_state[:, 0]
            logits2 = self.classifier(self.tanh(self.pooler(cls2)))
            loss2 = self.loss_func(logits2, labels.float())
            ret["loss"] += loss2
            ret["loss"] /= 2

        return ret


class BertModel_binary_classify_shift(BertModel):
    def __init__(self, config, alpha, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.alpha = alpha
        self.get_input_embeddings().weight.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, alpha, *model_args, **kwargs):
        kwargs["alpha"] = alpha
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss

        if is_train:
            input_embs = self.get_input_embeddings()(input_ids)
            input_embs = shift_embeddings(input_embs, self.alpha)
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_embs, output_attentions=False, output_hidden_states=False
            )
            cls2 = outputs2.last_hidden_state[:, 0]
            logits2 = self.classifier(self.tanh(self.pooler(cls2)))
            loss2 = self.loss_func(logits2, labels.float())
            ret["loss"] += loss2
            ret["loss"] /= 2

        return ret


# ! 有bug: infer的时候, 不应该加 noise的
class BertModel_binary_classify_noise_only(BertModel):
    def __init__(self, config, min_threshold, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.min_threshold = min_threshold

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, min_threshold=None, *model_args, **kwargs):
        kwargs["min_threshold"] = min_threshold
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        dis = generate_distribution(input_ids=input_ids, vocab_size=self.vocab_size, min_main_score=self.min_threshold)
        # emb = self.roberta.get_input_embeddings().weight.clone()
        emb = self.get_input_embeddings().weight
        input_emb = torch.matmul(dis, emb)
        outputs2 = super().forward(
            None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_emb, output_attentions=False, output_hidden_states=False
        )
        cls2 = outputs2.last_hidden_state[:, 0]
        logits2 = self.classifier(self.tanh(self.pooler(cls2)))
        ret["logits"] = logits2
        if labels is not None:
            loss2 = self.loss_func(logits2, labels.float())
            ret["loss"] = loss2
        return ret


# 不太行
###########################################################################################################################################


class RobertaModel_gaussion_label_bi(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 100)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        ret = dict()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
        # ret['weight_bert'] = outputs.attentions
        cls = outputs.last_hidden_state[:, 0]
        ret["cls_hidden_state"] = cls
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels)
        ret["loss"] = loss
        return ret


# 蒸馏相关
######################################################################################################################################################################


class Fusion(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.fusion_que_ans = nn.Linear(dim * 2, dim)
        self.fusion_qa = nn.Linear(dim * 2, dim)
        self.fusion_cls_qa = nn.Linear(dim * 2, dim)
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, cls, que1, ans1, que2, ans2):
        qa1 = self.gelu(self.fusion_que_ans(torch.cat((que1, ans1), dim=1)))
        qa2 = self.gelu(self.fusion_que_ans(torch.cat((que2, ans2), dim=1)))
        qas = self.tanh(self.fusion_qa(torch.cat((qa1, qa2), dim=1)))
        ret = self.tanh(self.fusion_cls_qa(torch.cat((cls, qas), dim=1)))
        return ret


# Roberta_student
class RobertaModel_student_bi(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

        self.fusion = Fusion(config.hidden_size)

    def forward(self, input_ids, attention_mask, cls_sep_idx, labels=None, **kwargs):
        ret = dict()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
        cls = outputs.last_hidden_state[:, 0]
        # pdb.set_trace()
        # for i in range(input_ids.shape[0]):
        #     print(outputs.last_hidden_state[i][cls_sep_idx[i][0] + 1 : cls_sep_idx[i][1] - 64].shape)
        que1 = torch.stack(
            [outputs.last_hidden_state[i][cls_sep_idx[i][0] + 1 : cls_sep_idx[i][1] - 64].mean(dim=0) for i in range(input_ids.shape[0])], dim=0
        )
        ans1 = torch.stack(
            [outputs.last_hidden_state[i][cls_sep_idx[i][1] - 64 : cls_sep_idx[i][1]].mean(dim=0) for i in range(input_ids.shape[0])], dim=0
        )
        que2 = torch.stack(
            [outputs.last_hidden_state[i][cls_sep_idx[i][2] + 1 : cls_sep_idx[i][-1] - 64].mean(dim=0) for i in range(input_ids.shape[0])], dim=0
        )
        ans2 = torch.stack(
            [outputs.last_hidden_state[i][cls_sep_idx[i][-1] - 64 : cls_sep_idx[i][-1]].mean(dim=0) for i in range(input_ids.shape[0])], dim=0
        )

        finalHiddenState = self.fusion(cls, que1, ans1, que2, ans2)
        logits = self.classifier(self.tanh(self.pooler(finalHiddenState)))
        ret["logits"] = logits
        ret["ans1"] = ans1
        ret["ans2"] = ans2

        if labels is None:
            return logits

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss
        return ret


# Roberta_teacher
class RobertaModel_teacher(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        # self.tanh = nn.Tanh()
        # self.classifier = nn.Linear(config.hidden_size, 1)
        # self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, cls_sep_idx, labels=None, **kwargs):
        ret = dict()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)

        ans1 = torch.stack([outputs.last_hidden_state[i][1 : cls_sep_idx[i][1]].mean(dim=0) for i in range(input_ids.shape[0])], dim=0)
        ans2 = torch.stack(
            [outputs.last_hidden_state[i][cls_sep_idx[i][2] + 1 : cls_sep_idx[i][-1]].mean(dim=0) for i in range(input_ids.shape[0])], dim=0
        )

        ret["ans1"] = ans1
        ret["ans2"] = ans2

        return ret


class Classifier(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, cls1, cls2):
        cls_fin = torch.concat([cls1, cls2], dim=-1)
        return self.classifier(self.tanh(self.pooler(cls_fin)))


# Roberta_ensemble_baseline
class RobertaModel_ensemble_bi(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = Classifier(config)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        ret = dict()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
        # ret['weight_bert'] = outputs.attentions
        cls = outputs.last_hidden_state[:, 0]
        ret["cls_hidden_state"] = cls
        return ret


# 第一个点
######################################################################################################################################################################


class Gate_and_fusion(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.fusion_same = nn.Linear(dim * 3, dim)
        self.fusion_dif = nn.Linear(dim * 2, dim)
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o):
        same = self.gelu(self.fusion_same(torch.cat((cls, sep_s1, sep_s2), dim=1)))
        cls_fin1 = self.tanh(self.fusion_dif(torch.cat((same, sep_s1_o), dim=1)))
        cls_fin2 = self.tanh(self.fusion_dif(torch.cat((same, sep_s2_o), dim=1)))
        return cls_fin1, cls_fin2


def log(x, n):
    n = torch.tensor(n)
    return torch.log(x) / torch.log(n)


def get_weights_loss(logits, labels, base):
    logits = logits.detach()
    labels = labels.detach()
    y = torch.sigmoid(logits)
    gama = 1 - 1 / base
    weight_loss = torch.ones((logits.shape[0], 1), device=logits.device)
    already_classable = (logits > 0).view(-1) == labels.bool()
    x = (abs(labels - y) * 2)[already_classable] * (gama)  # x: [game, 0]
    x = 1 - x
    weight_loss[already_classable] = -log(x, base)
    return weight_loss


class FixedPositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, max_len=512, num_hiddens=768, dropout=0.1):
        super().__init__()
        # self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return X
        # return self.dropout(X)


# class TrainablePositionalEncoding(nn.Module):
#     """位置编码"""
#     def __init__(self, seq_len=512, hidden_dim=768):
#         super().__init__()
#         self.weight = torch.zeros((seq_len, hidden_dim))
#         # self.P = torch.nn.Embedding(seq_len, hidden_dim)
#         X = torch.arange(seq_len, dtype=torch.float32).reshape(
#             -1, 1) / torch.pow(10000, torch.arange(
#             0, hidden_dim, 2, dtype=torch.float32) / hidden_dim)
#         self.weight[:, 0::2] = torch.sin(X)
#         self.weight[:, 1::2] = torch.cos(X)
#         self.P = torch.nn.Embedding(seq_len, hidden_dim).from_pretrained(self.weight)

#     def forward(self, X, position_ids):
#         X = X + self.P(position_ids)
#         return X


# # Roberta
# class RobertaMatchModel(RobertaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = RobertaModel(config)
#         self.pooler = nn.Linear(config.hidden_size*2, config.hidden_size)
#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
#         # --------------------------------------------------------------------------------------------------------------
#         self.fixed_position_encoder = FixedPositionalEncoding(config.max_position_embeddings, config.hidden_size)
#         # self.trainable_position_encoder = TrainablePositionalEncoding(config.max_position_embeddings, config.hidden_size)
#         # --------------------------------------------------------------------------------------------------------------
#         self.att_layer = WAIO(config, need_para=True)
#         self.fusion = Gate_and_fusion(config.hidden_size)


#     def forward(self, input_ids, attention_mask, cls_sep_idxs, labels=None):
#         outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         ret = dict()
#         # ret['representation'] = outputs.last_hidden_state
#         att_layer_input = outputs.last_hidden_state
#         # --------------------------------------------------------------------------------------------------------------
#         att_layer_input = self.fixed_position_encoder(att_layer_input)
#         # att_layer_input = self.trainable_position_encoder(att_layer_input, position_ids)
#         # --------------------------------------------------------------------------------------------------------------
#         last, last_overlook, weight, weight_o = self.att_layer(
#             att_layer_input, att_layer_input, att_layer_input,
#             key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs)

#         ret['weight'] = weight
#         ret['weight_o'] = weight_o
#         ret['weight_bert'] = outputs.attentions

#         cls = outputs.last_hidden_state[:, 0]  # shape: (batch, hidden_size)

#         # sep
#         sep_s1 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
#         sep_s1_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
#         sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
#         sep_s2_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
#         # --------------------------------------------------------------------------------------------------------------
#         cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
#         cls_fin = torch.cat((cls_fin1, cls_fin2), dim=1)
#         logits = self.classifier(self.tanh(self.pooler(cls_fin)))
#         # --------------------------------------------------------------------------------------------------------------

#         ret['logits'] = logits

#         if labels is None:
#             return ret

#         # 交叉熵损失函数
#         loss = self.loss_func(logits, labels.float())

#         # WAIO30
#         base = 2
#         weight_loss = get_weights_loss(logits, labels, base)
#         loss = torch.mm(loss.T, weight_loss)/loss.shape[0]


#         ret['loss'] = loss
#         return ret
# --------------------------------------------------------------------------------------------------------------

# # Roberta_without_mod1
# class RobertaMatchModel_without_mod1(RobertaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = RobertaModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, input_ids, attention_mask, labels=None, **kwargs):
#         ret = dict()
#         outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         # pooler_output = outputs[1]
#         # logits = self.linear(self.dropout(pooler_output))

#         cls = outputs.last_hidden_state[:, 0]
#         logits = self.classifier(self.tanh(self.pooler(cls)))
#         ret['logits'] = logits
#         if labels is None:
#             return logits
#         loss = self.loss_func(logits, labels.float())

#         # WAIO30
#         base = 1000
#         weight_loss = get_weights_loss(logits, labels, base)
#         loss = torch.mm(loss.T, weight_loss)/loss.shape[0]


#         ret['loss'] = loss
#         return ret

# # Roberta_withou_out_mod2
# class RobertaMatchModel_without_mod2(RobertaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = RobertaModel(config)
#         self.pooler = nn.Linear(config.hidden_size*2, config.hidden_size)
#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
#         # --------------------------------------------------------------------------------------------------------------
#         self.fixed_position_encoder = FixedPositionalEncoding(config.max_position_embeddings, config.hidden_size)
#         # self.trainable_position_encoder = TrainablePositionalEncoding(config.max_position_embeddings, config.hidden_size)
#         # --------------------------------------------------------------------------------------------------------------
#         self.att_layer = WAIO(config, need_para=True)
#         self.fusion = Gate_and_fusion(config.hidden_size)


#     def forward(self, input_ids, attention_mask, cls_sep_idxs, labels=None):
#         outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         ret = dict()
#         # ret['representation'] = outputs.last_hidden_state
#         att_layer_input = outputs.last_hidden_state
#         # --------------------------------------------------------------------------------------------------------------
#         att_layer_input = self.fixed_position_encoder(att_layer_input)
#         # att_layer_input = self.trainable_position_encoder(att_layer_input, position_ids)
#         # --------------------------------------------------------------------------------------------------------------
#         last, last_overlook, weight, weight_o = self.att_layer(
#             att_layer_input, att_layer_input, att_layer_input,
#             key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs)

#         ret['weight'] = weight
#         ret['weight_o'] = weight_o
#         ret['weight_bert'] = outputs.attentions

#         cls = outputs.last_hidden_state[:, 0]  # shape: (batch, hidden_size)

#         # sep
#         sep_s1 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
#         sep_s1_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
#         sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
#         sep_s2_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
#         # --------------------------------------------------------------------------------------------------------------
#         cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
#         cls_fin = torch.cat((cls_fin1, cls_fin2), dim=1)
#         logits = self.classifier(self.tanh(self.pooler(cls_fin)))
#         # --------------------------------------------------------------------------------------------------------------

#         ret['logits'] = logits

#         if labels is None:
#             return ret

#         # 交叉熵损失函数
#         loss = self.loss_func(logits, labels.float())

#         # # WAIO30
#         # base = 1000
#         # weight_loss = get_weights_loss(logits, labels, base)
#         # loss = torch.mm(loss.T, weight_loss)/loss.shape[0]


#         ret['loss'] = loss
#         return ret


class BertModelFirst(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert: BertModel = BertModel(config)
        # config.hidden_size=768
        # self.pooler = nn.Linear(config.hidden_size*3, config.hidden_size)
        self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")
        self.tanh = nn.Tanh()
        # --------------------------------------------------------------------------------------------------------------
        # self.my_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.fixed_position_encoder = FixedPositionalEncoding(config.max_position_embeddings, config.hidden_size)
        # self.trainable_position_encoder = TrainablePositionalEncoding(config.max_position_embeddings, config.hidden_size)
        # --------------------------------------------------------------------------------------------------------------
        self.att_layer = WAIO(config, need_para=True)

        self.fusion = Gate_and_fusion(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.loss_aux = nn.CosineEmbeddingLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, position_ids, cls_sep_idx, labels=None):
        # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
        # outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
        #                     position_ids=position_ids, output_attentions=True)
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_attentions=True)
        ret = dict()
        # ret['representation'] = outputs.last_hidden_state
        att_layer_input = outputs.last_hidden_state
        # --------------------------------------------------------------------------------------------------------------
        # att_layer_input = outputs.last_hidden_state+self.my_position_embeddings(position_ids)
        att_layer_input = self.fixed_position_encoder(att_layer_input)
        # att_layer_input = self.trainable_position_encoder(att_layer_input, position_ids)
        # --------------------------------------------------------------------------------------------------------------
        last, last_overlook, weight, weight_o = self.att_layer(
            att_layer_input, att_layer_input, att_layer_input, key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idx
        )

        # last_overlook += self.my_position_embeddings(position_ids)
        # last, last_overlook, weight, weight_o = self.att_layer(
        #     outputs.last_hidden_state, outputs.last_hidden_state, outputs.last_hidden_state,
        #     key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs)

        ret["weight"] = weight
        ret["weight_o"] = weight_o

        ret["weight_bert"] = outputs.attentions

        cls = outputs.last_hidden_state[:, 0]  # shape: (batch, hidden_size)
        # cls = last[:, 0]

        # sep
        sep_s1 = last[list(range(input_ids.shape[0])), cls_sep_idx[:, 1]]
        sep_s1_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idx[:, 1]]
        sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idx[:, 2]]
        sep_s2_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idx[:, 2]]
        # --------------------------------------------------------------------------------------------------------------
        cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
        cls_fin = torch.cat((cls_fin1, cls_fin2), dim=1)
        # same, cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
        # cls_fin = torch.cat((same, cls_fin1, cls_fin2), dim=1)
        # logits = self.classifier(self.dropout(self.tanh(self.pooler(cls_fin))))
        logits = self.classifier(self.tanh(self.pooler(cls_fin)))
        # --------------------------------------------------------------------------------------------------------------

        ret["logits"] = logits

        if labels is None:
            return ret

        # 交叉熵损失函数
        loss = self.loss_func(logits, labels.float())

        # 方法1 WAIO9
        # weight_loss = torch.softmax(torch.clone(loss).detach()*1, 0)
        # loss = torch.mm(loss.T, weight_loss)
        # # 方法2 WAIO10
        # alpha = 0  # 0
        # beta = 0.5  # 2
        # y = torch.sigmoid(logits).detach()
        # weight_loss = torch.ones((logits.shape[0], 1), device=logits.device)
        # already_classable = ((logits>0).view(-1)==labels)
        # weight_loss[already_classable] = ((abs(labels-y).detach()+alpha)[already_classable])*beta
        # loss = torch.mm(loss.T, weight_loss)/loss.shape[0]

        # WAIO30
        base = 2
        weight_loss = get_weights_loss(logits, labels, base)
        loss = torch.mm(loss.T, weight_loss) / loss.shape[0]

        # # 方法3 WAIO12
        # alpha = 0 # 0
        # beta = 1  # 2
        # y = torch.sigmoid(logits).detach()
        # weight_loss = (abs(labels-y).detach()+alpha)*beta
        # weight_loss = torch.softmax(weight_loss, 0)
        # loss = torch.mm(loss.T, weight_loss)/loss.shape[0]

        ret["loss"] = loss
        return ret


# # without_mod2
# class BertMatchModel_without_mod2(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert: BertModel = BertModel(config)
#         # config.hidden_size=768
#         self.pooler = nn.Linear(config.hidden_size*2, config.hidden_size)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
#         self.tanh = nn.Tanh()
#         # --------------------------------------------------------------------------------------------------------------
#         # self.my_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
#         self.fixed_position_encoder = FixedPositionalEncoding(config.max_position_embeddings, config.hidden_size)
#         # self.trainable_position_encoder = TrainablePositionalEncoding(config.max_position_embeddings, config.hidden_size)
#         # --------------------------------------------------------------------------------------------------------------
#         self.att_layer = WAIO(config, need_para=True)

#         self.fusion = Gate_and_fusion(config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#         # self.loss_aux = nn.CosineEmbeddingLoss()
#     def forward(self, input_ids, token_type_ids, attention_mask, position_ids, cls_sep_idxs, labels=None):
#         # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
#         # outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
#         #                     position_ids=position_ids, output_attentions=True)
#         outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
#                             output_attentions=True)
#         ret = dict()
#         # ret['representation'] = outputs.last_hidden_state
#         att_layer_input = outputs.last_hidden_state
#         # --------------------------------------------------------------------------------------------------------------
#         # att_layer_input = outputs.last_hidden_state+self.my_position_embeddings(position_ids)
#         att_layer_input = self.fixed_position_encoder(att_layer_input)
#         # att_layer_input = self.trainable_position_encoder(att_layer_input, position_ids)
#         # --------------------------------------------------------------------------------------------------------------
#         last, last_overlook, weight, weight_o = self.att_layer(
#             att_layer_input, att_layer_input, att_layer_input,
#             key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs)

#         # last_overlook += self.my_position_embeddings(position_ids)
#         # last, last_overlook, weight, weight_o = self.att_layer(
#         #     outputs.last_hidden_state, outputs.last_hidden_state, outputs.last_hidden_state,
#         #     key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs)

#         ret['weight'] = weight
#         ret['weight_o'] = weight_o

#         ret['weight_bert'] = outputs.attentions

#         cls = outputs.last_hidden_state[:, 0]  # shape: (batch, hidden_size)
#         # cls = last[:, 0]

#         # sep
#         sep_s1 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
#         sep_s1_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
#         sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
#         sep_s2_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]

#         cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
#         cls_fin = torch.cat((cls_fin1, cls_fin2), dim=1)

#         # logits = self.classifier(self.dropout(self.tanh(self.pooler(cls_fin))))
#         logits = self.classifier(self.tanh(self.pooler(cls_fin)))

#         ret['logits'] = logits

#         if labels is None:
#             return ret

#         # 交叉熵损失函数
#         loss = self.loss_func(logits, labels.float())

#         ret['loss'] = loss
#         return ret


# # without_mod1
# class BertMatchModel_without_mod1(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert: BertModel = BertModel(config)
#         # config.hidden_size=768
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.tanh = nn.Tanh()

#     def forward(self, input_ids, token_type_ids, attention_mask, labels=None, **kargs):
#         # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
#         outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
#                             output_attentions=True)

#         cls = outputs.last_hidden_state[:, 0]
#         # cls = outputs.pooler_output

#         cls = self.pooler(cls)
#         cls = self.tanh(cls)
#         logits = self.classifier(cls)

#         ret = {'logits': logits}

#         # ret['representation'] = outputs.last_hidden_state
#         ret['weight_bert'] = outputs.attentions
#         if labels is None:
#             return ret
#         # 交叉熵损失函数
#         loss = self.loss_func(logits, labels.float())

#         # without mod1
#         base = 1000
#         weight_loss = get_weights_loss(logits, labels, base)
#         loss = torch.mm(loss.T, weight_loss)/loss.shape[0]

#         ret['loss'] = loss
#         return ret


######################################################################################################################################################################
