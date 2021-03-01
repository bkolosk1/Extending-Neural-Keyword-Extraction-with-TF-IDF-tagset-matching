import copy
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torchcrf import CRF
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.activations import gelu, gelu_new, swish
from typing import Tuple
from torch import Tensor, device

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}

class CustomNLLLoss(torch.nn.Module):
    def __init__(self):
        super(CustomNLLLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, targets):
        logs = self.log_softmax(x)
        # out = torch.zeros_like(targets, dtype=torch.float)
        zeros = []
        ones = []
        for i in range(len(targets)):
            current = targets[i]

            if current == 1:
                ones.append(logs[i][targets[i]])

            elif current == 0:
                zeros.append(logs[i][targets[i]])

        if len(ones) > 0:
            r = -sum(ones) / len(ones)
        else:
            r = 0
        if len(zeros) > 0:
            p = -sum(zeros) / len(zeros)
        else:
            p = 0

        return p + r


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, config):
        super(BiLSTM_CRF, self).__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=1, bidirectional=True)
        self.NUM_TAGS = 2

        self.crf = CRF(self.NUM_TAGS)
        self.hidden2tag = nn.Linear(hidden_dim, self.NUM_TAGS)

    def init_hidden(self, batch_size):
        if not self.config.cuda:
            return (torch.randn(2, batch_size, self.hidden_dim),
                    torch.randn(2, batch_size, self.hidden_dim))
        else:
            return (torch.randn(2, batch_size, self.hidden_dim).cuda(),
                    torch.randn(2, batch_size, self.hidden_dim).cuda())

    def _get_lstm_features(self, embeds):
        self.hidden = self.init_hidden(embeds.size(0))
        embeds = embeds.permute(1, 0, 2)
        lstm_out, self.lstm_hidden = self.lstm(embeds, self.hidden)
        lstm_out = (lstm_out[:, :, :self.embedding_dim] + lstm_out[:, :, self.embedding_dim:])
        lstm_out = lstm_out.permute(1, 0, 2)
        embeds = embeds.permute(1, 0, 2)
        lstm_feats = self.hidden2tag(embeds + lstm_out)
        # print(lstm_feats.size())
        return lstm_feats

    def neg_log_likelihood(self, embeds, tags, mask=None):
        feats = self._get_lstm_features(embeds)
        feats = feats.permute(1, 0, 2)
        return -self.crf(feats, tags, mask=mask)

    def forward(self, embeds):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(embeds)
        lstm_feats = lstm_feats.permute(1, 0, 2)
        return lstm_feats, self.crf.decode(lstm_feats)


BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.n_ctx, config.n_embd)
        self.token_type_embeddings = nn.Embedding(200, config.n_embd)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.n_embd, eps=1e-12)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.n_embd, config.n_head)
            )

        self.num_attention_heads = config.n_head
        self.attention_head_size = int(config.n_embd / config.n_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.n_embd, self.all_head_size)
        self.key = nn.Linear(config.n_embd, self.all_head_size)
        self.value = nn.Linear(config.n_embd, self.all_head_size)

        self.dropout = nn.Dropout(0.3)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.LayerNorm = BertLayerNorm(config.n_embd, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, 3072)
        self.intermediate_act_fn = ACT2FN['gelu']


    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(3072, config.n_embd)
        self.LayerNorm = BertLayerNorm(config.n_embd, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = False
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.n_head)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class TransformerHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, config):
        super(TransformerHead, self).__init__()
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.div_val = 1
        self.sigmoid = nn.Sigmoid()

        self.adaptive_softmax = config.adaptive
        self.classification = config.classification
        self.masked_lm = config.masked_lm
        self.rnn = config.rnn
        self.crf = config.crf
        self.config = config

        if self.classification:
            self.classification = True
            self.loss_function = CustomNLLLoss()
            #self.loss_function = nn.CrossEntropyLoss()
            self.decoder = nn.Linear(self.config.n_embd, 2, bias=True)
            self.num_layers = 2

            if self.rnn:
                self.lstm = nn.LSTM(self.config.n_embd, self.config.n_embd, dropout=0.1, num_layers=self.num_layers,
                                    bidirectional=True)
                self.lstm_dropout = nn.Dropout(0.1)
            elif self.crf:
                self.crf = BiLSTM_CRF(config.vocab_size, config.n_embd, config.n_embd, config)

        else:
            self.loss_function = nn.CrossEntropyLoss()
            self.decoder = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=True)

        if self.adaptive_softmax:
            raise ValueError(
                "You can't have BERT with adaptive softmax"
            )

    def init_hidden(self, batch_size):
        if self.config.cuda:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.n_embd).cuda())
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.n_embd).cuda())
        else:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.n_embd))
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.n_embd))
        return (h0, c0)

    def forward(self, hidden_state, target, masked_idx, test=False, predict=False):

        if self.classification:
            if self.crf:
                mask = target.clone()
                mask[mask > 0] = 1
                mask = mask.type(torch.uint8)
                mask = mask.permute(1, 0)
                # print(target.cpu().numpy().tolist())
                target[target == 1] = 0
                target[target > 1] = 1
                target = target.permute(1, 0)

                if not test:
                    loss = self.crf.neg_log_likelihood(hidden_state, target, mask=mask)
                    # logits = self.crf._get_lstm_features(embeddings)
                    # loss = self.loss_function(logits.contiguous().view(-1, logits.size(-1)), target.contiguous().view(-1))
                    return loss
                elif predict:
                    logits, predictions = self.crf(hidden_state)
                    logits = logits.permute(1, 0, 2)
                    return logits, predictions
                elif test:
                    loss = self.crf.neg_log_likelihood(hidden_state, target, mask=mask)
                    logits, predictions = self.crf(hidden_state)
                    logits = logits.permute(1, 0, 2)
                    return loss, logits, predictions

                    # logits = self.crf._get_lstm_features(embeddings)
                    # loss = self.loss_function(logits.contiguous().view(-1, logits.size(-1)), target.contiguous().view(-1))
                    # return loss, logits, logits

            else:
                if self.rnn:
                    lstm_hidden = self.init_hidden(hidden_state.size(0))
                    embeddings = self.lstm_dropout(hidden_state).permute(1, 0, 2)
                    lstm_out, self.lstm_hidden = self.lstm(embeddings, lstm_hidden)
                    lstm_out = (lstm_out[:, :, :self.n_embd] + lstm_out[:, :, self.n_embd:])
                    lstm_out = lstm_out.permute(1, 0, 2)
                    hidden_state = hidden_state + lstm_out

                logits = self.decoder(self.dropout(self.relu(hidden_state)))
                if predict:
                    return logits

                active_loss = target.contiguous().view(-1) > 0
                active_logits = logits.contiguous().view(-1, logits.size(-1)).squeeze(1)[active_loss]
                # active_multi_logits = multi_logits.contiguous().view(-1, multi_logits.size(-1)).squeeze(1)[active_loss]

                active_targets = target.contiguous().view(-1) - 1

                active_targets = active_targets[active_loss]

                binary_targets = active_targets.clone()
                binary_targets[binary_targets > 0] = 1

                binary_loss = self.loss_function(active_logits, binary_targets)

                loss = binary_loss

                if test:
                    return loss, logits
                return loss

        else:
            # use adaptive softmax (including standard softmax)
            if self.adaptive_softmax:
                if test:
                    logits = self.dropout(self.decoder(hidden_state))

                # hidden_state = hidden_state.permute(1, 0, 2)
                # target = target.permute(1, 0)

                if self.masked_lm:
                    hidden_state = hidden_state[masked_idx]
                    target = target[masked_idx]

                loss = self.crit(hidden_state.contiguous().view(-1, hidden_state.contiguous().size(-1)),
                                 target.contiguous().view(-1), test)
            else:
                logits = self.dropout(self.decoder(hidden_state))
                loss = self.loss_function(logits.contiguous().view(-1, logits.size(-1)), target.contiguous().view(-1))
            if test:
                return loss, logits
            return loss

class BertModel(nn.Module):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super(BertModel, self).__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.head = TransformerHead(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple, device: device) -> Tensor:

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask: Tensor, num_hidden_layers: int, is_attention_chunked: bool = False) -> Tensor:

        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask


    def forward(self, input_ids, input_pos=None, lm_labels=None, embeddings=None, masked_idx=None, test=False, predict=False):
        if input_ids is not None:
            input_shape = input_ids.size()

        param = next(self.parameters())

        if masked_idx is None:
            attention_mask = torch.ones(input_shape, device=param.device)
        if input_pos is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=param.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, param.device)
        encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(None, self.config.n_head)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None
        )
        hidden_states, att = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=True,
            output_hidden_states=None,
        )

        if predict and self.config.classification:
            logits = self.head(hidden_states, lm_labels, masked_idx, predict=predict)
            return logits

        if test:
            if self.config.classification:
                if self.config.crf:
                    loss, logits, embedding_logits = self.head(hidden_states, lm_labels, masked_idx, test=test)
                    # att = torch.diagonal(att, dim1=-2, dim2=-1)
                    return loss, logits, embedding_logits, att
                else:
                    loss, logits = self.head(hidden_states, lm_labels, masked_idx, test=test)
                    # att = torch.diagonal(att, dim1=-2, dim2=-1)
                    return loss, logits, att
            else:
                loss, logits = self.head(hidden_states, lm_labels, masked_idx, test=test)
                return loss, logits
        loss = self.head(hidden_states, lm_labels, masked_idx, test=test)

        return loss









