from transformers import BertModel, GPT2Model, BertForMaskedLM
from .transformer import CustomNLLLoss, BiLSTM_CRF
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            #self.loss_function =  CustomNLLLoss()
            self.loss_function = CrossEntropyLoss()
            self.decoder = nn.Linear(self.config.n_embd, 2, bias=True)
            self.num_layers = 2

            if self.rnn:
                self.lstm = nn.LSTM(self.config.n_embd, self.config.n_embd, dropout=0.3, num_layers=self.num_layers, bidirectional=True)
                self.lstm_dropout = nn.Dropout(0.3)
            elif self.crf:
                self.crf = BiLSTM_CRF(config.vocab_size, config.n_embd, config.n_embd, config)



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
                mask = mask.permute(1,0)
                #print(target.cpu().numpy().tolist())
                target[target == 1] = 0
                target[target > 1] = 1
                target = target.permute(1,0)

                if not test:
                    loss = self.crf.neg_log_likelihood(hidden_state, target, mask=mask)
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

            else:
                if self.rnn:
                    lstm_hidden = self.init_hidden(hidden_state.size(0))
                    embeddings = self.lstm_dropout(hidden_state).permute(1, 0, 2)
                    #print(embeddings.size())
                    lstm_out, self.lstm_hidden = self.lstm(embeddings, lstm_hidden)
                    lstm_out = (lstm_out[:, :, :self.n_embd] + lstm_out[:, :, self.n_embd:])
                    lstm_out = lstm_out.permute(1,0,2)
                    hidden_state = hidden_state + lstm_out

                logits = self.decoder(self.dropout(self.relu(hidden_state)))
                if predict:
                    return logits

                active_loss = target.contiguous().view(-1) > 0
                active_logits = logits.contiguous().view(-1, logits.size(-1)).squeeze(1)[active_loss]
                #active_multi_logits = multi_logits.contiguous().view(-1, multi_logits.size(-1)).squeeze(1)[active_loss]

                active_targets = target.contiguous().view(-1) - 1

                active_targets = active_targets[active_loss]

                binary_targets = active_targets.clone()
                binary_targets[binary_targets > 0] = 1
                #print("Active logits: ", active_logits)
                #print("binary targets: ", binary_targets)

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

                #hidden_state = hidden_state.permute(1, 0, 2)
                #target = target.permute(1, 0)

                if self.masked_lm:
                    hidden_state = hidden_state[masked_idx]
                    target = target[masked_idx]

                loss = self.crit(hidden_state.contiguous().view(-1, hidden_state.contiguous().size(-1)), target.contiguous().view(-1), test)
            else:
                logits = self.dropout(self.decoder(hidden_state))
                loss = self.loss_function(logits.contiguous().view(-1, logits.size(-1)), target.contiguous().view(-1))

            if test:
                return loss, logits
            return loss


class BertClassifier(nn.Module):

    def __init__(self, config):
        super(BertClassifier, self).__init__()

        self.config = config
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        #self.bert = GPT2Model.from_pretrained('gpt2')
        #for param in self.bert.parameters():
        #    param.requires_grad = False
        self.head = TransformerHead(config)

    def forward(self, input_ids, input_pos=None, lm_labels=None, masked_idx=None, test=False,
                    predict=False):
        mask = lm_labels.clone()
        mask[mask > 0] = 1
        mask = mask.type(torch.uint8)
        hidden_states , _ = self.bert(input_ids=input_ids, attention_mask=mask)

        #print("STATES: ", hidden_states.size(), "IE: ", inputs_embeds.size(), len(att))

        if predict and self.config.classification:
            logits = self.head(hidden_states, lm_labels, masked_idx, predict=predict)
            return logits

        if test:
            if self.config.classification:
                if self.config.crf:
                    loss, logits, embedding_logits = self.head(hidden_states, lm_labels, masked_idx,
                                                               test=test)
                    # att = torch.diagonal(att, dim1=-2, dim2=-1)
                    return loss, logits, embedding_logits
                else:
                    loss, logits = self.head(hidden_states, lm_labels, masked_idx, test=test)
                    # att = torch.diagonal(att, dim1=-2, dim2=-1)
                    return loss, logits
            else:
                loss, logits = self.head(hidden_states, lm_labels, masked_idx, test=test)
                return loss, logits
        loss = self.head(hidden_states, lm_labels, masked_idx, test=test)

        return loss









