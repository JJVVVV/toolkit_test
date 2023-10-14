from typing import Optional, Tuple, Union

from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.configuration_t5 import T5Config


class T5Model(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

    def forward(
        self,
        input_ids: LongTensor | None = None,
        attention_mask: FloatTensor | None = None,
        decoder_input_ids: LongTensor | None = None,
        decoder_attention_mask: BoolTensor | None = None,
        head_mask: FloatTensor | None = None,
        decoder_head_mask: FloatTensor | None = None,
        cross_attn_head_mask: Tensor | None = None,
        encoder_outputs: Tuple[Tuple[Tensor]] | None = None,
        past_key_values: Tuple[Tuple[Tensor]] | None = None,
        inputs_embeds: FloatTensor | None = None,
        decoder_inputs_embeds: FloatTensor | None = None,
        labels: LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        is_train: bool = True,
    ) -> Tuple[FloatTensor] | Seq2SeqLMOutput:
        return super().forward(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
