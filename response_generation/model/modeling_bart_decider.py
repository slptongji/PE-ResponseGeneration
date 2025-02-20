# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BART model. """
import copy
import math
import random
from typing import Optional, Tuple
from model.Decoder import Decoder
from model.Encoder import Encoder
import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from transformers import BartPretrainedModel, BartConfig
from transformers.models.bart.modeling_bart import (
    BartLearnedPositionalEmbedding,
    BartEncoderLayer,
    BartDecoderLayer,
    _expand_mask,
    _make_causal_mask,
    shift_tokens_right,
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)

from transformers.file_utils import (
    add_code_sample_docstrings,
    replace_return_docstrings,
)

from transformers.utils import logging
from model.PriorNet import PriorNet
from model.RecognizeNet import RecognizeNet



logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"
emo_dictionary = {'anger':0, 'disgust':1, 'fear':2, 'joy':3, 'neutral':4, 'sadness':5, 'surprise':6}


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # See all BART models at https://huggingface.co/models?filter=bart
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        # print("device here")
        # print(input_ids.device)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)



        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        latent_memory=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        if latent_memory != None:
            assert latent_memory.size(-1) == inputs_embeds.size(-1)
            print("shape check hereee")
            print(input_ids.size())
            print(hidden_states[:,0,:].size())
            print(latent_memory.size())
            hidden_states[:,0,:] = hidden_states[:,0,:] + latent_memory

        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        "The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )



class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )



class BartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past



class BartForSequenceClassification(BartPretrainedModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )



class BartForQuestionAnswering(BartPretrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = BartModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.model._init_weights(self.qa_outputs)

    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        start_positions=None,
        end_positions=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if start_positions is not None and end_positions is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class BartDecoderWrapper(BartPretrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the :class:`~transformers.EncoderDecoderModel` framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = BartDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class BartForCausalLM(BartPretrainedModel):
    def __init__(self, config):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = BartDecoderWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two
                additional tensors are only required when the model is used as a decoder in a Sequence to Sequence
                model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last ``decoder_input_ids``
                (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
                instead of all ``decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
                config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are
                ignored (masked), the loss is only computed for the tokens with labels in ``[0, ...,
                config.vocab_size]``.
            use_cache (:obj:`bool`, `optional`):
                If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
                decoding (see :obj:`past_key_values`).

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.

        Returns:

        Example::

            >>> from transformers import BartTokenizer, BartForCausalLM

            >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            >>> model = BartForCausalLM.from_pretrained('facebook/bart-large', add_cross_attention=False)
            >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> logits = outputs.logits
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past:
            input_ids = input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


from model.memory_loss import MemoryLoss

class GRUEncoder(nn.Module):
    def __init__(self, emb_dim, rnn_hidden_dim, sent_dim, bigru, dropout=0.3):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(emb_dim, rnn_hidden_dim, num_layers=1, bidirectional=bigru)
        self.proj = nn.Linear(2 * rnn_hidden_dim, sent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sent, sent_len):
        # (N, L, D_w) -> (L, N, D_w)
        sent_embs = sent.transpose(0, 1)

        # padding
        # (L, N, D_w) -> (L, N, 2*D_h)
        sent_packed = pack_padded_sequence(sent_embs, sent_len, enforce_sorted=False)

        sent_output, h_n = self.gru(sent_packed)
        sent_output = pad_packed_sequence(sent_output, total_length=sent.size(1))[0]

        # (L, N, 2*D_h) -> (N, L, 2*D_h)
        sent_output = sent_output.transpose(0, 1)

        # # max poolingsent.size(1)
        # # (N, L, 2*D_h) -> (N, 2*D_h, L) ->
        # # (N, 2*D_h, 1) -> (N, 1, 2*D_h)
        # maxpout = F.max_pool1d(sent_output.transpose(2, 1), sent_output.size(1))
        # maxpout = maxpout.transpose(2, 1)
        #
        # # (N, 1, 2*D_h) -> (N, 1, D_s) -> (N, D_s)
        # sent_rep = self.dropout(F.relu(self.proj(maxpout)))
        # sent_rep = sent_rep.squeeze(1)
        dim1 = h_n.size()[1]
        return sent_output, h_n.transpose(0, 1).reshape(dim1, -1)

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(
            self,
            input_depth,
            total_key_depth,
            total_value_depth,
            output_depth,
            num_heads,
            bias_mask=None,
            dropout=0.0,
    ):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

        if total_key_depth % num_heads != 0:
            print(
                "Key depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_key_depth, num_heads)
            )
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print(
                "Value depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_value_depth, num_heads)
            )
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5  ## sqrt
        self.bias_mask = bias_mask

        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(
            shape[0], shape[1], self.num_heads, shape[2] // self.num_heads
        ).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (
            x.permute(0, 2, 1, 3)
                .contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads)
        )

    def forward(self, queries, keys, values, mask):

        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)

        ## attention weights
        attetion_weights = logits.sum(dim=1) / self.num_heads

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)

        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)

        # Merge heads
        contexts = self._merge_heads(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)

        return outputs, attetion_weights


class LMEDRModel(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, num_token=None, num_latent=20, num_latent2=5, num_latent3=10,emo_num_labels = 7):
        super().__init__(config)
        self.model = BartModel(config)
        # self.model = torch.nn.DataParallel(self.model,[0,1]).module
        self.num_latent = num_latent
        self.num_latent2 = num_latent2
        self.num_latent3 = num_latent3
        self.emo_num_labels = emo_num_labels
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        if num_token != None:
            self.bow_head = nn.Linear(config.d_model, num_token)
        else:
            self.bow_head = nn.Linear(config.d_model, self.model.config.vocab_size)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        self.emo_classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.emo_num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

        self.latent_head_m1 = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.num_latent,
            config.classifier_dropout,
        )
        self.model._init_weights(self.latent_head_m1.dense)
        self.model._init_weights(self.latent_head_m1.out_proj)

        self.latent_head_m2 = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.num_latent2,
            config.classifier_dropout,
        )
        self.model._init_weights(self.latent_head_m2.dense)
        self.model._init_weights(self.latent_head_m2.out_proj)

        self.latent_head_m3 = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.num_latent3,
            config.classifier_dropout,
        )
        self.model._init_weights(self.latent_head_m3.dense)
        self.model._init_weights(self.latent_head_m3.out_proj)

        # 用于情感编码的Bi-LSTM
        self.bi_gru = True
        self.num_hidd = 2 if self.bi_gru else 1
        decoder_number= len(emo_dictionary)
        self.gru = GRUEncoder(emb_dim=config.d_model, rnn_hidden_dim=config.d_model,
                              sent_dim=config.d_model, bigru=self.bi_gru)
        self.attention = Attention(config.d_model, 2 * config.d_model)
        self.tagging = nn.Linear(self.num_hidd * config.d_model, decoder_number)
        self.dia_attention = Attention(config.d_model, 2 * config.d_model)
        self.dia_att_lin = nn.Linear(2 * config.d_model, self.emo_num_labels)
        self.emotion_embedding = nn.Linear(decoder_number, config.d_model)
        self.PAD_idx = 1
        self.depth = 40
        self.heads =2
        self.emo_know = MultiHeadAttention(
            input_depth=config.d_model,
            total_key_depth=self.depth,
            total_value_depth=self.depth,
            output_depth=config.d_model,
            num_heads=self.heads,
        )

        self.dmodel = config.d_model
        self.persona_norm = nn.LayerNorm(config.d_model, elementwise_affine=True)  # 512
        self.memory1 = nn.Parameter(torch.randn(self.num_latent, config.d_model))
        self.memory2 = nn.Parameter(torch.randn(self.num_latent2, config.d_model))
        self.memory3 = nn.Parameter(torch.randn(self.num_latent3, config.d_model))

        self.encoder_2 = Encoder()

        self.prior_net_p_list = []
        self.recognize_net_p_list = []

        for _ in range(config.emo_N):
            self.prior_net_p_list.append(PriorNet(768,
                                  768,
                                  config.dims_recognize))
            self.recognize_net_p_list.append(RecognizeNet(768,
                                          768,
                                          768,
                                          config.dims_recognize))

        self.prior_net = PriorNet(768,  # The input dimension
                                  768,  # Latent variable dimension
                                  config.dims_prior)  # Dimensions of hidden layers

        # recognition network-r
        self.recognize_net = RecognizeNet(768,
                                          768,
                                          768,
                                          config.dims_recognize)

        self.decider = nn.Sequential(
            nn.Linear(768 * config.emo_N + 768, 768),
            nn.Tanh(),
            nn.Linear(768, config.emo_N + 1))

        self.decoder_2 = Decoder()
        self.post_init()
        self.emo_N = config.emo_N

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def mean_pooling(self, encoder_outputs, mask_src):

        mask = mask_src.eq(False).float()

        encode = mask * encoder_outputs  # 把padd部分置0
        summ = encode.sum(dim=-2)  # 在第一维L上求和b,1
        lenth = mask.sum(dim=-2) + 1e-30  # 求出有多长 B,1
        en = summ / lenth
        return en
    # def history_split(self):

    def history_split(self, emo_words_list_input,history, history_emo, history_emo_input_ids, res_id, sep_id, persona_id,pad_id):
        his_s = []
        his_emo_s = []
        his_emo_i_s = []
        emo_words_list_s = []
        emo_words_list_attention = []

        for emo_words,his,his_emo,his_emo_i in zip(emo_words_list_input,history,history_emo,history_emo_input_ids):
            #截取掉填充
            sub_s,sub_emo_s,sub_emo_i_s,sub_emo_words_s=[],[],[],[]

            indices_p = torch.nonzero(his == res_id).squeeze()
            indices_emo_p = torch.nonzero(his_emo == res_id).squeeze()
            indices_emo_i_p = torch.nonzero(his_emo_i == res_id).squeeze()
            indices_emo_words_i_p = torch.nonzero(emo_words == res_id).squeeze()

            his = his[:indices_p.item()]
            # print(his)
            his_emo = his_emo[:indices_emo_p.item()]
            # print(his_emo)
            his_emo_i = his_emo_i[:indices_emo_i_p.item()]
            emo_words = emo_words[:indices_emo_words_i_p.item()]


            indices = torch.nonzero(his == persona_id).squeeze()
            indices_emo = torch.nonzero(his_emo == persona_id).squeeze()
            # indices_emo_i是直接的串联
            indices_emo_i = torch.nonzero(his_emo_i == persona_id).squeeze()
            # indices_emo_words_i是把每个单词都分开
            indices_emo_words_i = torch.nonzero(emo_words == persona_id).squeeze()
            # print("item check")
            # print(indices.item())
            his_s.append(his[0:indices.item()])
            his_emo_s.append(his_emo[0:indices_emo.item()])

            his_emo_i_s.append(his_emo_i[0:indices_emo_i.item()])

            prev_index=0
            emo_words_list_s_u = []
            for index in indices_emo_words_i:
                emo_words_list_s.append(emo_words[prev_index:index])
                prev_index = index + 1
            emo_words_list_s.append(emo_words[prev_index:])
            # emo_words_list_attention_u = [[1] * len(hs) for hs in emo_words_list_s_u]
            # emo_words_list_attention_u = pad_sequence([torch.from_numpy(np.array(x)) for x in emo_words_list_attention_u],
            #                                         batch_first=True, padding_value=0).to(torch.cuda.current_device())
            #
            # emo_words_list_s_u = pad_sequence(emo_words_list_s_u, batch_first=True, padding_value=pad_id)
            # emo_words_list_s.append(emo_words_list_s_u)
            # emo_words_list_attention.append(emo_words_list_attention_u)
            # emo_words_list_s.append(emo_words[0:indices_emo_words_i.item()])

            # attention的

        his_s_attention = [[1]*len(hs) for hs in his_s]
        his_s_emo_attention = [[1] * len(hs) for hs in his_emo_s]
        his_s_emo_i_attention = [[1] * len(hs) for hs in his_emo_i_s]
        emo_words_list_attention = [[1] * len(hs) for hs in emo_words_list_s]



        his_s = pad_sequence(his_s,batch_first=True, padding_value=pad_id)
        # print(his_s.size())
        his_emo_s = pad_sequence(his_emo_s,batch_first=True, padding_value=pad_id)
        his_emo_i_s = pad_sequence(his_emo_i_s,batch_first=True, padding_value=pad_id)
        emo_words_list_s = pad_sequence(emo_words_list_s, batch_first=True, padding_value=pad_id)

        emo_words_list_s = emo_words_list_s.view(his_emo_s.size()[0],-1,emo_words_list_s.size()[-1])



        his_s_attention = pad_sequence([torch.from_numpy(np.array(x)) for x in his_s_attention],batch_first=True, padding_value=0).to(torch.cuda.current_device())
        his_s_emo_attention = pad_sequence([torch.from_numpy(np.array(x)) for x in his_s_emo_attention], batch_first=True, padding_value=0).to(torch.cuda.current_device())
        his_s_emo_i_attention = pad_sequence([torch.from_numpy(np.array(x)) for x in his_s_emo_i_attention], batch_first=True, padding_value=0).to(torch.cuda.current_device())
        emo_words_list_attention = pad_sequence([torch.from_numpy(np.array(x)) for x in emo_words_list_attention], batch_first=True, padding_value=0).to(torch.cuda.current_device())

        emo_words_list_attention = emo_words_list_attention.view(his_s_attention.size()[0],-1,emo_words_list_attention.size()[-1])

        # emo_words_list_attention = pad_sequence(emo_words_list_attention, batch_first=True, padding_value=pad_id)


        return his_s,his_emo_s,his_emo_i_s,emo_words_list_s,his_s_attention,his_s_emo_attention,his_s_emo_i_attention,emo_words_list_attention

    def emo_vec(self, his_hidden,his_emo_labels,his_emo_i_hidden, emolabels,res_id, sep_id,persona_id):
        # 加入seq_length这个维度
        his_s = his_hidden.view(his_hidden.size()[0], 1, -1)  # (bs, 1, hidden_dim)
        his_s_length = [1] * (his_s.size()[0])
        lstm_output, lstm_h_n = self.gru(his_s, sent_len=his_s_length)
        lstm_mask = torch.zeros(lstm_output.size()[0], lstm_output.size()[1]).to(torch.cuda.current_device())
        for idx, leng in enumerate(his_s_length):
            lstm_mask[idx, :leng] = torch.Tensor([1] * leng)
        a = self.attention(lstm_h_n, lstm_output, lstm_mask).unsqueeze(1)
        atten_output = torch.bmm(a, lstm_output).squeeze(1)
        utt_emo_logits = self.tagging(lstm_output)
        # 计算出回复中表达的情感类别
        emo_pred = self.tagging(atten_output)
        # print("emo_pred!")
        # print(utt_emo_logits)

        # 计算每个utt中的细粒度情感损失
        utt_emo_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)(
            utt_emo_logits.reshape(utt_emo_logits.size()[0] * utt_emo_logits.size()[1], -1),
            his_emo_labels.reshape(-1))

        # 计算生成回复中的情感表达损失
        if emolabels is not None:
            trg_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)(emo_pred.squeeze(1), emolabels)
        else:
            trg_loss = None

        # cuda_d = torch.cuda.current_device()
        emo_emb = self.emotion_embedding(emo_pred).to(torch.cuda.current_device())
        ctx_emo = self.emotion_embedding(utt_emo_logits).to(torch.cuda.current_device())

        his_emo_i_s = his_emo_i_hidden.view(his_emo_i_hidden.size()[0], 1, -1)
        atten_outputs, attetion_weights = self.emo_know(ctx_emo, ctx_emo, ctx_emo, None)
        atten_outputs = self.mean_pooling(atten_outputs, atten_outputs.data.eq(self.PAD_idx)).unsqueeze(1)
        sos_emb = torch.cat((emo_emb.unsqueeze(1), atten_outputs), dim=1)
        sos_emb = self.mean_pooling(sos_emb, sos_emb.data.eq(self.PAD_idx))


        return utt_emo_loss, trg_loss, sos_emb
    def cal_z_p_matrix(self,  emo_tokens, per_tokens, en_tokens, de_tokens, en_de_tokens):
        # total_emo_lists:[bs,emo_N,length,embedding]
        # per_tokens = " ".join(per_tokens)
        # en_tokens = " ".join(en_tokens)
        # de_tokens = " ".join(de_tokens)


        _mu_p_list = []
        _logvar_p_list = []
        mu_p_list = []
        logvar_p_list = []
        z_p_list = []

        encoder_outputs_2 = self.encoder_2(en_tokens)
        # print("emo outputs")
        # print(emo_tokens)
        # print(len(emo_tokens))
        # print(len(emo_tokens[0]))
        # print(len(emo_tokens[0][0]))
        decoder_outputs_2 = self.encoder_2(de_tokens)
        emo_token_group = len(emo_tokens)
        emo_token_group_len = len(emo_tokens[0])
        emo_token_g = np.array(emo_tokens).reshape(-1)
        # print(emo_token_g.shape)
        emo_outputs_2 = self.encoder_2(emo_token_g.tolist())
        emo_outputs_2 = emo_outputs_2.view(emo_token_group,emo_token_group_len,-1)
        # print(emo_outputs_2.size())
        # emo_outputs_2 =



        sampled_latents = torch.randn(encoder_outputs_2.size()[0], encoder_outputs_2.size()[-1]).to(encoder_outputs_2.device)
        z_p_zero = torch.zeros((sampled_latents.shape[0], encoder_outputs_2.size()[-1])).to(encoder_outputs_2.device)
        for i in range(self.emo_N):
            # print(encoder_outputs.device)
            _mu_p, _logvar_p = self.prior_net_p_list[i](encoder_outputs_2)

            mu_p, logvar_p = self.recognize_net_p_list[i](encoder_outputs_2, emo_outputs_2[:, i, :])  # [batch,latent]
            # print("nup[")
            # print(mu_p.size())
            z_p = mu_p + (0.5 * logvar_p).exp() * sampled_latents  # [batch, latent]
            _mu_p_list.append(_mu_p)
            _logvar_p_list.append(_logvar_p)
            mu_p_list.append(mu_p)
            logvar_p_list.append(logvar_p)
            z_p_list.append(z_p)

        z_p_matrix = torch.stack(z_p_list + [z_p_zero])
        z_p_matrix = z_p_matrix.permute(1, 0, 2)
        decide_matrix = self.decider(torch.cat(z_p_list + [encoder_outputs_2], 1)).unsqueeze(1)

        decide_matrix_softmax = torch.softmax(decide_matrix, dim=2)
        z_p_w = torch.bmm(decide_matrix_softmax, z_p_matrix).squeeze(1)
        z_p = z_p_w

        _mu, _logvar = self.prior_net(encoder_outputs_2)  # [batch, latent]
        # p(z|q,r)
        mu, logvar = self.recognize_net(encoder_outputs_2, decoder_outputs_2)  # [batch, latent]
        # parameterized

        z_r = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]


        torch.no_grad()
        candidate_loss = []
        eos_token = '<|endoftext|>'
        for i in range(z_p_matrix.shape[1]):
            candidate_loss_batch = []
            for j in range(z_p_matrix.shape[0]):
                # print(per_tokens.device)

                candidate_loss_batch.append(
                    self.decoder_2(True, [per_tokens[j]+eos_token], [en_tokens[j]+eos_token], [de_tokens[j]+eos_token],
                                 [en_de_tokens[j]+eos_token], z_r[j].unsqueeze(0).to("cuda:0"), z_p_matrix[j, i, :].unsqueeze(0).to("cuda:0")))
            candidate_loss.append(torch.stack(candidate_loss_batch))

        # [N+1,B]
        candidate_loss = torch.stack(candidate_loss)
        # Loss to the minimum indexes
        loss_index = torch.argmin(candidate_loss, dim=0)

        # The cross entropy loss
        decide_loss_fct = nn.CrossEntropyLoss()

        decide_loss = decide_loss_fct(decide_matrix.squeeze(1), loss_index)

        # decode_loss = self.decoder_2(True, text_persons, encoder_outputs, decoder_outputs, text_posts_responses, z_r, z_p)
        return decide_loss

        #
        # print("decide_matrix!")
        # print(decide_matrix.size())



    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            input_ids_2=None,
            # attention_mask_2=None,
            en_de_input_ids = None,
            # en_de_attention_mask = None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_input_ids_2=None,
            # decoder_attention_mask_2=None,

            # en_de_input_id = None,
            # en_de_attention_mask_2 = None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            lmlabels=None,
            clslabel=None,
            cls_index=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            infer_input_ids=None,
            infer_decoder_input_ids=None,
            infer_attention_mask=None,
            infer_decoder_attention_mask=None,
            infer_lmlabels=None,
            per_input_ids=None,
            per_attention_mask=None,
            per_input_ids_n=None,
            # per_attention_mask_n=None,
            emo_input_ids=None,
            emo_attention_mask=None,
            emolabel=None,
            history=None,
            history_emo=None,
            history_emo_input_ids=None,
            emo_words_list_input = None,
            return_dict=True,
            generate=False,
            latent_variable=None,
            infer=True,
            dialog=True,
            emo=True,
            atten=False,
            emo_vec=False,
            res_id=None,
            sep_id=None,
            persona_id=None,
            pad_id=None,
            tokenizer_2=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """

        global encoder_outputs_
        # print("global")
        # print(encoder_outputs_)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if infer_lmlabels is None:
            self.memory1.requires_grad = False



        if input_ids != None:
            bs = input_ids.size(0)
            input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask != None:
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        if input_ids_2 != None:
            input_ids_2 = input_ids_2.view(-1, input_ids_2.size(-1))
        # if attention_mask_2 != None:
        #     attention_mask_2 = attention_mask_2.view(-1, attention_mask_2.size(-1))
        if en_de_input_ids != None:
            en_de_input_ids = en_de_input_ids.view(-1, en_de_input_ids.size(-1))
        # if en_de_attention_mask != None:
        #     en_de_attention_mask = en_de_attention_mask.view(-1, en_de_attention_mask.size(-1))
        if decoder_input_ids != None:
            bs = decoder_input_ids.size(0)
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size(-1))
        if decoder_attention_mask != None:
            decoder_attention_mask =decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))
        if decoder_input_ids_2 != None:
            decoder_input_ids_2 = decoder_input_ids_2.view(-1, decoder_input_ids_2.size(-1))
        # if decoder_attention_mask_2 != None:
        #     decoder_attention_mask_2 =decoder_attention_mask_2.view(-1, decoder_attention_mask_2.size(-1))
        if cls_index != None:
            cls_index = cls_index.view(-1,cls_index.size(-1))
        if per_input_ids != None:
            per_input_ids = per_input_ids.view(-1, per_input_ids.size(-1))
        if per_attention_mask != None:
            per_attention_mask = per_attention_mask.view(-1, per_attention_mask.size(-1))
        if per_input_ids_n != None:
            per_input_ids_n = per_input_ids_n.view(-1, per_input_ids_n.size(-1))
        # if per_attention_mask_n != None:
        #     per_attention_mask_n = per_attention_mask_n.view(-1, per_attention_mask_n.size(-1))
        # 加入情感的处理
        if emo_input_ids != None:
            emo_input_ids = emo_input_ids.view(-1, emo_input_ids.size(-1))
        if emo_attention_mask != None:
            emo_attention_mask = emo_attention_mask.view(-1,emo_attention_mask.size(-1))
        if history != None:
            history = history.view(-1, history.size(-1))
        if history_emo != None:
            history_emo = history_emo.view(-1,history_emo.size(-1))
        if history_emo_input_ids != None:
            history_emo_input_ids = history_emo_input_ids.view(-1,history_emo_input_ids.size(-1))

        if emo_words_list_input != None:
            emo_words_list_input = emo_words_list_input.view(-1,emo_words_list_input.size(-1))

        if infer_lmlabels != None:
            infer_lmlabels = infer_lmlabels.view(-1, infer_lmlabels.size(-1))

        if infer_attention_mask != None:
            infer_attention_mask = infer_attention_mask.view(-1, infer_attention_mask.size(-1))

        if infer_decoder_input_ids != None:
            infer_decoder_input_ids = infer_decoder_input_ids.view(-1, infer_decoder_input_ids.size(-1))

        if infer_decoder_attention_mask != None:
            infer_decoder_attention_mask = infer_decoder_attention_mask.view(-1, infer_decoder_attention_mask.size(-1))

        if infer_input_ids != None:
            infer_input_ids = infer_input_ids.view(-1, infer_input_ids.size(-1))



        print("reagch gere")
        # print(input_ids)
        infer_masked_lm_loss = None
        if infer:
            if infer_lmlabels is not None:
                infer_encoder_outputs = self.model.encoder(
                    input_ids=infer_input_ids,
                    attention_mask=infer_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                infer_latent_hidden_state = infer_encoder_outputs.last_hidden_state[:, 0, :]
                infer_latent_logits = self.latent_head_m1(infer_latent_hidden_state)

                weight_memory = torch.mm(torch.softmax(infer_latent_logits, dim=-1), self.memory1)

                infer_decoder_outputs = self.model.decoder(
                    input_ids=infer_decoder_input_ids,
                    attention_mask=infer_decoder_attention_mask,
                    encoder_hidden_states=infer_encoder_outputs[0],
                    encoder_attention_mask=infer_attention_mask,
                    head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    latent_memory=weight_memory
                )

                infer_lm_logits = self.lm_head(infer_decoder_outputs[0]) + self.final_logits_bias
                loss_fct = CrossEntropyLoss()
                infer_masked_lm_loss = loss_fct(infer_lm_logits.view(-1, self.config.vocab_size), infer_lmlabels.view(-1))


        if lmlabels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    lmlabels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        dialog_latent_variable = None
        entail_latent_variable = None
        emo_latent_variable = None
        utt_emo_loss = None
        trg_loss = None
        if input_ids is not None:
            # if infer:
            # print("input_ids is not None")
            # print(input_ids)
            encoder_per_outputs = self.model.encoder(
                    input_ids=per_input_ids,
                    attention_mask=per_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )


            latent_hidden_state_m1 = encoder_per_outputs.last_hidden_state[:, 0, :]
            last_hidden_state_1 = encoder_per_outputs.last_hidden_state


            latent_logits_m1 = self.latent_head_m1(latent_hidden_state_m1)
            entail_latent_variable = torch.mm(torch.softmax(latent_logits_m1, dim=-1), self.memory1)

            encoder_outputs_ = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # print("result check")
            # print(encoder_outputs_)


            # latent_hidden_state_m2 = encoder_outputs.last_hidden_state[:, 0, :]
            last_hidden_state_2 = encoder_outputs_.last_hidden_state
            # print("hidden state size")
            # print(encoder_outputs[0].size())
            # print(encoder_outputs.last_hidden_state.size())
            # latent_mul = torch.stack([torch.matmul(i, j.t()) for i, j in
            #                        zip(last_hidden_state_1, last_hidden_state_2)])
            # print(latent_mul)
            # 不使用latent_hidden_state_m2是因为它取的是最后一个隐藏状态的第一个位置，也就是[z]
            # 但这里要用的是整个句子的隐藏状态
            latent_hidden_state_d_p = torch.stack([torch.matmul(torch.softmax(torch.matmul(i, j.t()), dim=-1), j) for i, j in
                                   zip(last_hidden_state_1, last_hidden_state_2)])
            # 更新后的latent_hidden_state_m2
            latent_hidden_state_d_p = self.persona_norm(latent_hidden_state_d_p+last_hidden_state_1)



            if history is not None and atten:
                # 形状都是(batch_size,length)
                his_s, his_emo_labels, history_emo_input_words, emo_words_list_s, his_s_attention, his_s_emo_attention, his_s_emo_i_attention, emo_words_list_attention=self.history_split(
                    emo_words_list_input,history, history_emo, history_emo_input_ids, res_id, sep_id, persona_id, pad_id)
                encode_his_outputs = self.model.encoder(
                    input_ids=his_s,
                    attention_mask=his_s_attention,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                encode_his_emo_i_outputs = self.model.encoder(
                    input_ids=history_emo_input_words,
                    attention_mask=his_s_emo_i_attention,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                # print(history_emo_input_words.size())
                encode_emo_words_list_l = []
                emo_encode_dict = {}
                # print(emo_words_list_s.size())
                for i in range(emo_words_list_s.size()[0]):
                    emo_encode_dict[i]=[]
                # emo_encode_dict={"0":[],"1":[],"2":[],"3":[],"4":[],"5":[]}
                for i in range(emo_words_list_s.size()[1]):
                    emo_tensor = emo_words_list_s[:,i,:]
                    # print("emo_tensor!")
                    # print(emo_tensor.size())
                    emo_attrn_tensor = emo_words_list_attention[:,i,:]
                    encode_emo_words_list_s = self.model.encoder(
                            input_ids=emo_tensor,
                            attention_mask=emo_attrn_tensor,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                        ).last_hidden_state # [6,3,1024]
                    for j in range(encode_emo_words_list_s.size()[0]):
                        emo_encode_dict[j].append(encode_emo_words_list_s[j][-1,:])
                for i in range(emo_words_list_s.size()[0]):
                    # print()
                    emo_encode_dict[i] = torch.cat(emo_encode_dict[i],dim=0).view(emo_words_list_s.size()[1],-1)

                total_emo_lists = []
                for i in range(emo_words_list_s.size()[0]):
                    total_emo_lists.append(emo_encode_dict[i])
                total_emo_lists = torch.cat(total_emo_lists,dim=0).view(emo_words_list_s.size()[0],emo_words_list_s.size()[1],-1)
                # print(emo_words_list_s.size())
                # print(emo_words_list_s)
                # print(encoder_outputs_.to_tensor())
                # print("encoder decoder check")
                # print(per_input_ids_n.size())
                # print(en_de_input_ids.size())
                if decoder_input_ids_2 is not None:
                    print("decoder_input_ids_2 not None")
                    emo_tokens,per_tokens, en_tokens, de_tokens, en_de_tokens = [],[],[],[],[]
                    eos_token = '<|endoftext|>'

                    # if per_input_ids_n == None:
                    #     print("per_input_ids_n None here!")
                    for emo_tensor,per_tensor, en_tensor, de_tensor, en_de_tensor in zip(emo_words_list_s,per_input_ids_n,input_ids_2,decoder_input_ids_2,en_de_input_ids):
                        emo_tensor = emo_tensor.tolist()
                        per_tensor = per_tensor.tolist()
                        en_tensor = en_tensor.tolist()
                        de_tensor = de_tensor.tolist()
                        en_de_tensor = en_de_tensor.tolist()
                        emo_tokens_t = []
                        for emo_t in emo_tensor:
                            # emo_t = emo_t.tolist()
                            test_p = [str(inpp) for inpp in emo_t if inpp != 2 and inpp != 1]
                            emo_tokens_t.append(tokenizer_2.decode(test_p))
                        emo_tokens.append(emo_tokens_t)
                        # print(emo_tokens)
                        # emo_tokens.append(tokenizer.decode([str(inpp) for inpp in emo_tensor if inpp != 2 and inpp != 1]))
                        per_tokens.append(tokenizer_2.decode([str(inpp) for inpp in per_tensor if inpp is not None]).rstrip('"\''))
                        en_tokens.append(tokenizer_2.decode([str(inpp) for inpp in en_tensor if inpp is not None]).rstrip('"\''))
                        de_tokens.append(tokenizer_2.decode([str(inpp) for inpp in de_tensor if inpp is not None]).rstrip('"\''))
                        en_de_tokens.append(tokenizer_2.decode([str(inpp) for inpp in en_de_tensor if inpp is not None]).rstrip('"\''))
                        # print(tokenizer.decode([str(inpp) for inpp in inp if inpp is not None]))
                    # print(emo_tokens)
                    #
                    # print(de_tokens)
                    # print(en_de_tokens)
                    # print(input_ids_2[0].tolist())
                    # inp = input_ids_2[0].tolist()
                    # # inp = [3,1,3,2,4,78]
                    # print(type(inp))
                    # print(tokenizer.decode([str(inpp) for inpp in inp if inpp is not None]))
                    # encoder_outputs_2 = self.model.encoder(
                    #     input_ids=input_ids_2,
                    #     attention_mask=attention_mask_2,
                    #     head_mask=head_mask,
                    #     inputs_embeds=inputs_embeds,
                    #     output_attentions=output_attentions,
                    #     output_hidden_states=output_hidden_states,
                    #     return_dict=return_dict,
                    # )
                    #
                    # decoder_outputs_2 = self.model.encoder(
                    #     input_ids=decoder_input_ids_2,
                    #     attention_mask=decoder_attention_mask_2,
                    #     head_mask=head_mask,
                    #     inputs_embeds=inputs_embeds,
                    #     output_attentions=output_attentions,
                    #     output_hidden_states=output_hidden_states,
                    #     return_dict=return_dict,
                    # )
                    # en_de_outputs_2 = self.model.encoder(
                    #     input_ids=en_de_input_ids,
                    #     attention_mask=en_de_attention_mask,
                    #     head_mask=head_mask,
                    #     inputs_embeds=inputs_embeds,
                    #     output_attentions=output_attentions,
                    #     output_hidden_states=output_hidden_states,
                    #     return_dict=return_dict,
                    # )

                    # print("decoder output")
                    # print(en_de_outputs_2.last_hidden_state.size())
                    decide_loss = self.cal_z_p_matrix( emo_tokens,per_tokens,en_tokens,de_tokens,en_de_tokens)

                his_hidden = encode_his_outputs.last_hidden_state[:, 0, :]
                his_emo_i_hidden = encode_his_emo_i_outputs.last_hidden_state[:, 0, :]
                his_emo_i_last_hidden = encode_his_emo_i_outputs.last_hidden_state




                latent_hidden_state_d = torch.stack(
                    [torch.matmul(torch.softmax(torch.matmul(i, j.t()), dim=-1), j) for i, j in
                     zip(latent_hidden_state_d_p, his_emo_i_last_hidden)])
                # 更新后的latent_hidden_state_m2
                latent_hidden_state_m2 = self.persona_norm(latent_hidden_state_d + latent_hidden_state_d_p)

                # print(latent_hidden_state_m2.size())

                if dialog:
                    latent_logits_m2 = self.latent_head_m2(latent_hidden_state_m2[:, 0, :])
                    dialog_latent_variable = torch.mm(torch.softmax(latent_logits_m2, dim=-1), self.memory2)

                # if emolabel:
                utt_emo_loss, trg_loss, emo_emb = self.emo_vec(his_hidden, his_emo_labels, his_emo_i_hidden, emolabel,
                                                              res_id, sep_id, persona_id)

            if emo:
                encode_emo_outputs = self.model.encoder(
                    input_ids = emo_input_ids,
                    attention_mask = emo_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                # latent_hidden_state_m3 = encode_emo_outputs.last_hidden_state[:, 0 , :]
                # 对偶过程，融入persona和dialogue

                # 融合persona
                last_hidden_state_3 = encode_emo_outputs.last_hidden_state
                latent_hidden_state_e_p = torch.stack(
                    [torch.matmul(torch.softmax(torch.matmul(i, j.t()), dim=-1), j) for i, j in
                     zip(last_hidden_state_3, last_hidden_state_1)])
                # 更新后的latent_hidden_state_m2
                latent_hidden_state_e_p = self.persona_norm(latent_hidden_state_e_p + last_hidden_state_3)

                # 融合对话
                latent_hidden_state_m3 = torch.stack(
                    [torch.matmul(torch.softmax(torch.matmul(i, j.t()), dim=-1), j) for i, j in
                     zip(latent_hidden_state_e_p, last_hidden_state_2)])
                # 更新后的latent_hidden_state_m2
                latent_hidden_state_m3 = self.persona_norm(latent_hidden_state_m3 + latent_hidden_state_e_p)

                latent_logits_m3 = self.latent_head_m3(latent_hidden_state_m3[:,0,:])
                emo_latent_variable = torch.mm(torch.softmax(latent_logits_m3,dim=-1), self.memory3) #[batch_size,d_model]
                # print("zundujiadu")
                # print(emo_latent_variable.size())
                if history is not None and emo_vec:
                    emo_latent_variable = emo_emb #[batch_size,d_model]


            if generate:
                if infer and dialog and emo:
                    return {"latent_variable": dialog_latent_variable + entail_latent_variable + emo_latent_variable, "encoder_outputs": encoder_outputs_}
                elif not dialog:
                    return {"latent_variable": entail_latent_variable + emo_latent_variable,  "encoder_outputs": encoder_outputs_}
                elif not infer:
                    return {"latent_variable": dialog_latent_variable + emo_latent_variable, "encoder_outputs": encoder_outputs_}
                elif not emo:
                    return {"latent_variable": entail_latent_variable + dialog_latent_variable, "encoder_outputs": encoder_outputs_}





        if decoder_input_ids is not None:
            if latent_variable is not None:
                print("reach 1 here")
                input_latent = latent_variable
            elif dialog_latent_variable is not None and entail_latent_variable is not None and emo_latent_variable is not None:
                print("reach 2 here")
                input_latent = dialog_latent_variable + entail_latent_variable + emo_latent_variable
            elif entail_latent_variable is not None and dialog_latent_variable is None and emo_latent_variable is None:
                print("reach 3 here")
                input_latent = entail_latent_variable
            elif dialog_latent_variable is not None and entail_latent_variable is None and emo_latent_variable is None:
                print("reach 4 here")
                input_latent = dialog_latent_variable
            elif  entail_latent_variable is not None and dialog_latent_variable is not None and emo_latent_variable is None:
                print("reach 5 here")
                input_latent = entail_latent_variable + dialog_latent_variable
            elif entail_latent_variable is not None and dialog_latent_variable is None and emo_latent_variable is not None:
                print("reach 6 here")
                input_latent = entail_latent_variable + emo_latent_variable
            elif entail_latent_variable is None and dialog_latent_variable is not None and emo_latent_variable is not None:
                print("reach 7 here")
                input_latent = dialog_latent_variable + emo_latent_variable
            else:
                print("reach 8 here")
                input_latent = None
            print("decoder_input_ids check ")
            print(decoder_input_ids)
            decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=encoder_outputs_[0],
                    encoder_attention_mask=attention_mask,
                    head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    latent_memory=input_latent,
                )

            lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias
            if input_ids is None:
                lm_logits = lm_logits.view(bs, -1, lm_logits.size(-1))
                return Seq2SeqLMOutput(
                    loss=None,
                    logits=lm_logits,
                    past_key_values=decoder_outputs.past_key_values,
                    decoder_hidden_states=decoder_outputs.last_hidden_state,
                    decoder_attentions=decoder_outputs.attentions,
                    cross_attentions=decoder_outputs.cross_attentions,
                    encoder_last_hidden_state=encoder_outputs_.last_hidden_state,
                    encoder_hidden_states=encoder_outputs_.hidden_states,
                    encoder_attentions=encoder_outputs_.attentions,
                )
            seq_len = decoder_input_ids.size(-1)
            bow_logits = self.bow_head(input_latent).repeat(seq_len, 1, 1).transpose(0,1).contiguous() #for notdialog
            hidden_states = decoder_outputs[0]  # last hidden state




        masked_lm_loss = None
        bow_loss = None
        if lmlabels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lmlabels.view(-1))
            bow_loss = loss_fct(bow_logits.view(-1, self.config.vocab_size), lmlabels.view(-1))
            lm_logits = lm_logits.view(bs, -1, lm_logits.size(-1))


        cls_logits = None
        emo_logits = None
        if cls_index != None:
            cls_mask = cls_index.eq(self.config.eos_token_id)

            if len(torch.unique_consecutive(cls_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation = hidden_states[cls_mask, :].view(hidden_states.size(0), -1,
                                                                      hidden_states.size(-1))[:, -1, :]
            cls_logits = self.classification_head(sentence_representation)
            cls_logits = cls_logits.view(bs, -1)
            emo_logits = self.emo_classification_head(sentence_representation)
            emo_logits = emo_logits.view(bs, -1, self.emo_num_labels)[:,0,:]
            # cls_sfx = torch.argmax(cls_logits, dim=1).tolist()
            # print(cls_sfx.tolist())
            # emo_logits_l = []
            # for i in range(len(cls_sfx)):
            #     emo_logits_l.append(torch.tensor(torch.unsqueeze(emo_logits[i][cls_sfx[i]], 0)))
            # emo_logits = torch.cat(emo_logits_l, dim=0)

        cls_loss = None
        if clslabel is not None:
            loss_fct = CrossEntropyLoss()
            cls_loss = loss_fct(cls_logits, clslabel.view(-1))

        emo_loss = None
        if emolabel is not None:
            loss_fct = CrossEntropyLoss()
                # 并不需要cand，只需要batch_size,emo_num就可以，因为错误的答案反而是干扰项
            emo_loss = loss_fct(emo_logits, emolabel.view(-1))


        m_loss = None
        m_loss_2 = None

        if input_ids is not None and infer and dialog:
            m_fct = MemoryLoss()
            m_loss = m_fct(self.memory1, self.memory2)

        if input_ids is not None and infer and emo:
            m_fct_2 = MemoryLoss()
            m_loss_2 = m_fct_2(self.memory1, self.memory3)

        if input_ids == None:
            # print("input id none")
            masked_lm_loss = torch.tensor(0)
            cls_loss =torch.tensor(0)
            emo_loss=torch.tensor(0)
            m_loss =torch.tensor(0)
            m_loss_2=torch.tensor(0)
            bow_loss =torch.tensor(0)
            utt_emo_loss =torch.tensor(0)
            trg_loss = torch.tensor(0)
            decide_loss = torch.tensor(0)
            return Seq2SeqLMOutput(
                loss=infer_masked_lm_loss+masked_lm_loss+cls_loss+emo_loss+m_loss+m_loss_2+bow_loss+utt_emo_loss+trg_loss + decide_loss
            )
        else:
            # infer_masked_lm_loss被收集了，但是并没有被trainer处理，是为了给infer处理的
            infer_masked_lm_loss = torch.tensor(0)
            return Seq2SeqLMOutput(
                loss=(masked_lm_loss, cls_loss,emo_loss, m_loss, m_loss_2, infer_masked_lm_loss, bow_loss, utt_emo_loss,trg_loss,decide_loss),
                logits=(lm_logits, emo_logits),
            )


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        latent_variable=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used

        if past is not None:

            decoder_input_ids = decoder_input_ids[:, -1:]

        if past is None:
            return {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "past_key_values": past,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask,
                "head_mask": head_mask,
                "decoder_head_mask": decoder_head_mask,
                "cross_attn_head_mask": cross_attn_head_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
                "latent_variable": latent_variable,
            }
        else:
            return {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "past_key_values": past,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask,
                "head_mask": head_mask,
                "decoder_head_mask": decoder_head_mask,
                "cross_attn_head_mask": cross_attn_head_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
