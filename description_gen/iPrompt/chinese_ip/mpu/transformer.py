# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer."""

import math

import torch
import torch.nn.init as init
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .initialize import get_model_parallel_world_size
from .layers import ColumnParallelLinear
from .layers import RowParallelLinear
from .mappings import gather_from_model_parallel_region

import deepspeed

from .random import checkpoint
from .random import get_cuda_rng_tracker

from .utils import divide
from .utils import split_tensor_along_last_dim


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, hidden_size):
        super(PositionalEmbedding, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class GPT2ParallelSelfAttention(torch.nn.Module):
    """Parallel self-attention layer for GPT2.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """
    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, output_layer_init_method=None, relative_encoding=False):
        super(GPT2ParallelSelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)
        self.relative_encoding = relative_encoding
        # Strided linear layer.
        self.query_key_value = ColumnParallelLinear(hidden_size, 3*hidden_size,
                                                    stride=3,
                                                    gather_output=False,
                                                    init_method=init_method)
        if relative_encoding:
            self.relative = ColumnParallelLinear(hidden_size, hidden_size, gather_output=False,
                                                 init_method=init_method)
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def _rel_shift(x, zero_triu=False):
        # ql x kl x bsz x h
        # bsz x h x ql x kl
        zero_pad = torch.zeros((*x.size()[:-2], x.size(-2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:-2], x.size(-1) + 1, x.size(-2))

        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    @staticmethod
    def _rel_shift_latest(x: torch.Tensor):
        ndims = x.dim()
        x_shape = x.size()
        row_dim = 2
        col_dim = row_dim + 1
        assert col_dim < ndims
        tgt_shape_1, tgt_shape_2 = [], []
        for i in range(ndims):
            if i == row_dim:
                tgt_shape_1.append(x_shape[col_dim])
                tgt_shape_2.append(x_shape[row_dim])
            elif i == col_dim:
                tgt_shape_1.append(x_shape[row_dim])
                tgt_shape_2.append(x_shape[col_dim] - 1)
            else:
                tgt_shape_1.append(x_shape[i])
                tgt_shape_2.append(x_shape[i])
        x = x.view(*tgt_shape_1)
        x = x[:, :, 1:, :]
        x = x.view(*tgt_shape_2)
        return x

    def forward(self, hidden_states, ltor_mask, position_embeddings=None, r_w_bias=None, r_r_bias=None, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Attention heads. [b, s, hp]
        query_length = hidden_states.size(1)

        if mem is None:
            mixed_x_layer = self.query_key_value(hidden_states)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            cat = torch.cat((mem, hidden_states), 1)
            mixed_x_layer = self.query_key_value(cat)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
            mixed_query_layer = mixed_query_layer[:, -query_length:]

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        if self.relative_encoding:
            relative_layer = self.relative(position_embeddings)
            relative_layer = self._transpose_for_scores(relative_layer)  # 1 (bsz) x n_head x klen x d_head
            # Raw attention scores. [b, np, qs, ks]
            rw_head_q = query_layer + r_w_bias.unsqueeze(1)
            ac_score = torch.matmul(rw_head_q, key_layer.transpose(-1, -2))
            rr_head_q = query_layer + r_r_bias.unsqueeze(1)
            bd_score = torch.matmul(rr_head_q, relative_layer.transpose(-1, -2))
            bd_score = self._rel_shift(bd_score)  # qlen x klen x bsz x n_head
            # bd_score = bd_score.permute(2, 3, 0, 1) # bsz n_head qlen klen

            attention_scores = ac_score + bd_score
        else:
            # Raw attention scores. [b, np, s, s]
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.hidden_size_per_attention_head)
        # Apply the left to right attention mask.
        attention_scores = torch.mul(attention_scores, ltor_mask) - \
                           10000.0 * (1.0 - ltor_mask)

        # Attention probabilities. [b, np, s, s]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


@torch.jit.script
def gelu_impl(x):
     """OpenAI's gelu implementation."""
     return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                        (1.0 + 0.044715 * x * x)))

def gelu(x):
    return gelu_impl(x)


class GPT2ParallelMLP(torch.nn.Module):
    """MLP for GPT2.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    def __init__(self, hidden_size, output_dropout_prob, init_method,
                 output_layer_init_method=None):
        super(GPT2ParallelMLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4*hidden_size,
                                                  gather_output=False,
                                                  init_method=init_method)
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4*hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method)
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class GPT2ParallelTransformerLayer(torch.nn.Module):
    """A single layer transformer for GPT2.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None,
                 relative_encoding=False):
        super(GPT2ParallelTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = torch.nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = GPT2ParallelSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            relative_encoding=relative_encoding)

        # Layernorm on the input data.
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_size,
                                                  eps=layernorm_epsilon)

        # MLP
        self.mlp = GPT2ParallelMLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(self, hidden_states, ltor_mask, position_embeddings=None, r_w_bias=None, r_r_bias=None, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        mem = self.input_layernorm(mem) if mem is not None else None
        # Self attention.
        attention_output = self.attention(layernorm_output, ltor_mask, position_embeddings, r_w_bias, r_r_bias, mem)
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return output


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GPT2ParallelTransformer(torch.nn.Module):
    """GPT-2 transformer.

    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """
    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 max_memory_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 use_scaled_init_for_output_weights=True,
                 relative_encoding=False):
        super(GPT2ParallelTransformer, self).__init__()
        # Store activation checkpoiting flag.
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_memory_length = max_memory_length

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(init_method_std,
                                                      num_layers)
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.relative_encoding = relative_encoding
        if relative_encoding:
            # Relative position embedding
            self.position_embeddings = PositionalEmbedding(hidden_size)
            # Per attention head and per partition values.
            world_size = get_model_parallel_world_size()
            self.hidden_size_per_attention_head = divide(hidden_size,
                                                         num_attention_heads)
            self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                            world_size)
            self.r_w_bias = torch.nn.Parameter(
                torch.Tensor(self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
            self.r_w_bias.model_parallel = True
            self.r_r_bias = torch.nn.Parameter(
                torch.Tensor(self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
            self.r_r_bias.model_parallel = True
            # Always initialize bias to zero.
            with torch.no_grad():
                self.r_w_bias.zero_()
                self.r_r_bias.zero_()
        else:
            # Position embedding (serial).
            self.position_embeddings = torch.nn.Embedding(max_sequence_length,
                                                          hidden_size)
            # Initialize the position embeddings.
            torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        def get_layer():
            return GPT2ParallelTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                unscaled_init_method(init_method_std),
                output_layer_init_method=output_layer_init_method,
                relative_encoding=relative_encoding)

        # Transformer layers.
        self.layers = torch.nn.ModuleList(
            [get_layer() for _ in range(num_layers)])

        # Final layer norm before output.
        self.final_layernorm = torch.nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def forward(self, hidden_states, position_ids, attention_mask, *mems):
        batch_size, query_length = hidden_states.size()[:2]
        memory_length = mems[0].size(1) if mems else 0
        key_length = query_length + memory_length
        attention_mask = attention_mask[:, :, :, -query_length - memory_length:]
        if self.relative_encoding:
            hidden_states = self.embedding_dropout(hidden_states)
            position_sequence = torch.arange(key_length - 1, -1, -1.0, device=hidden_states.device,
                                             dtype=hidden_states.dtype)
            position_embeddings = self.position_embeddings(position_sequence)
            # Apply dropout
            position_embeddings = self.embedding_dropout(position_embeddings)
            hidden_states = self.embedding_dropout(hidden_states)
        else:
            position_embeddings = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeddings
            hidden_states = self.embedding_dropout(hidden_states)
        if self.max_memory_length > 0:
            mem_layers = [hidden_states.detach()]
        else:
            mem_layers = []
        def custom(start, end):
            def custom_forward(*inputs):
                layers_ = self.layers[start:end]
                x_, inputs = inputs[0], inputs[1:]
                if self.relative_encoding:
                    inputs, mems_ = inputs[:4], inputs[4:]
                else:
                    inputs, mems_ = inputs[:1], inputs[1:]
                for i, layer in enumerate(layers_):
                    mem_i_ = mems_[i] if mems_ else None
                    x_ = layer(x_, *inputs, mem=mem_i_)
                    if self.max_memory_length > 0:
                        mem_layers.append(x_.detach())
                return x_
            return custom_forward

        if self.checkpoint_activations:
            l = 0
            num_layers = len(self.layers)
            chunk_length = self.checkpoint_num_layers
            while l < num_layers:
                args = [hidden_states, attention_mask]
                if self.relative_encoding:
                    args += [position_embeddings, self.r_w_bias, self.r_r_bias]
                if mems:
                    args += mems[l: l + chunk_length]
                hidden_states = checkpoint(custom(l, l + chunk_length), *args)
                l += chunk_length
        else:
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask]
                if self.relative_encoding:
                    args += [position_embeddings, self.r_w_bias, self.r_r_bias]
                mem_i = mems[i] if mems else None
                hidden_states = layer(*args, mem=mem_i)
                if self.max_memory_length > 0:
                    mem_layers.append(hidden_states.detach())

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if self.max_memory_length > 0:
            mem_layers = self.update_mems(mem_layers, mems)

        return (output, *mem_layers)

    def update_mems(self, hiddens, mems):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = min(self.max_memory_length, memory_length + query_length)
        new_mems = []
        with torch.no_grad():
            for i in range(len(hiddens)):
                if new_memory_length <= query_length:
                    new_mems.append(hiddens[i][:, -new_memory_length:])
                else:
                    new_mems.append(torch.cat((mems[i][:, -new_memory_length+query_length:], hiddens[i]), dim=1))
        return new_mems


class BertParallelSelfAttention(torch.nn.Module):
    """Parallel self-attention layer for BERT.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size be divisible by n.
        dropout_prob: dropout probability for the attention scores.
        output_parallel: If true, no all-gather is done on the output and
                         the output values will be per partition.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """
    def __init__(self, hidden_size, num_attention_heads,
                 dropout_prob, output_parallel=False,
                 init_method=init.xavier_normal_):
        super(BertParallelSelfAttention, self).__init__()
        # Input configuration.
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob
        self.output_parallel = output_parallel
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)
        # Strided linear layer.
        self.query_key_value = ColumnParallelLinear(hidden_size, 3*hidden_size,
                                                    stride=3,
                                                    gather_output=False,
                                                    init_method=init_method)
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.dropout = torch.nn.Dropout(dropout_prob)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint


    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):

        # Attention heads. [b, s, hp]
        mixed_x_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # Raw attention scores. [b, np, s, s]
        norm_factor = math.sqrt(math.sqrt(self.hidden_size_per_attention_head))
        attention_scores = torch.matmul(query_layer/norm_factor,
                                        key_layer.transpose(-1, -2)/norm_factor)
        # Apply the attention mask.
        attention_scores += attention_mask

        # Attention probabilities. [b, np, s, s]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with get_cuda_rng_tracker().fork():
            attention_probs = self.dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        if self.output_parallel:
            output = context_layer
        else:
            output = gather_from_model_parallel_region(context_layer)

        return output


class BertParallelTransformerOutput(torch.nn.Module):
    """The output layer used after self attention and intermediate
    parts of transformer layer."""
    def __init__(self, input_size, output_size, dropout_prob,
                 layernorm_epsilon=1.0e-12, input_is_parallel=False,
                 init_method=init.xavier_normal_):
        super(BertParallelTransformerOutput, self).__init__()
        # Components.
        self.dense = RowParallelLinear(input_size,
                                       output_size,
                                       input_is_parallel=input_is_parallel,
                                       init_method=init_method)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.layernorm = torch.nn.LayerNorm(output_size, eps=layernorm_epsilon)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        layernorm_input = hidden_states + input_tensor
        hidden_states = self.layernorm(layernorm_input)
        return hidden_states


class BertParallelTransformerLayer(torch.nn.Module):
    """A single layer transformer for Bert.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        intermediate_size: size of the intermediate state after
                           self attention. In both BERT and GPT
                           this is set to be 4 times the hidden
                           size.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        intermediate_activation_fn: activation function for output
                                    of intermediate.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
    """
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 intermediate_activation_fn,
                 layernorm_epsilon,
                 init_method=init.xavier_normal_):
        super(BertParallelTransformerLayer, self).__init__()

        # Self attention.
        self.attention = BertParallelSelfAttention(hidden_size,
                                                   num_attention_heads,
                                                   attention_dropout_prob,
                                                   output_parallel=True,
                                                   init_method=init_method)
        # Self attention output.
        self.self_output = BertParallelTransformerOutput(
            hidden_size, hidden_size, output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            input_is_parallel=True,
            init_method=init_method)
        # Intermediate.
        self.intermediate = ColumnParallelLinear(hidden_size, intermediate_size,
                                                 gather_output=False,
                                                 init_method=init_method)
        self.intermediate_activation_fn = intermediate_activation_fn
        # Output.
        self.output = BertParallelTransformerOutput(
            intermediate_size, hidden_size, output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            input_is_parallel=True,
            init_method=init_method)

    def forward(self, hidden_states, attention_mask):
        # [b, s, hp]
        attention_output_parallel = self.attention(hidden_states,
                                                   attention_mask)
        # [b, s, h]
        attention_self_output = self.self_output(attention_output_parallel,
                                                 hidden_states)
        # [b, s, ip]
        intermediate_output_parallel = self.intermediate(attention_self_output)
        intermediate_output_parallel = self.intermediate_activation_fn(
            intermediate_output_parallel)
        # [b, s, h]
        layer_output = self.output(intermediate_output_parallel,
                                   attention_self_output)

        return layer_output
