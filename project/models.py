from pytools import memoize_method
import torch
import torch.nn.functional as F
import transformers
import modeling_util


class BertRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.CHANNELS = 12 + 1
        self.BERT_SIZE = 768
        self.bert = CustomBertModel.from_pretrained("bert-base-multilingual-cased")
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 3  # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        # execute BERT model
        result = self.bert(toks, segment_ids.long(), mask)

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN + 1] for r in result]
        doc_results = [r[:, QLEN + 2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results


class MT5Ranker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mt5 = transformers.MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
        self.tokenizer = transformers.MT5Tokenizer.from_pretrained("google/mt5-base")
        self.vocab = self.tokenizer.get_vocab()

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        return toks

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, labels):
        BATCH, QLEN = query_tok.shape
        DIFF = 2  # 2x</s>
        maxlen = 512
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        labels = torch.cat([labels] * sbcount, dim=0)

        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.eos_token_id)
        ONES = torch.ones_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([query_mask, ONES, doc_mask, ONES], dim=1)
        labels = torch.cat([labels, SEPS], dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        # execute BERT model
        result = self.mt5(input_ids=toks, attention_mask=mask, labels=labels)

        cls_output = result.logits[:, 0]
        cls_result = []
        for i in range(cls_output.shape[0] // BATCH):
            cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
        cls_result = torch.stack(cls_result, dim=2).mean(dim=2)

        return result.loss, cls_result

    def generate(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 2  # 2x</s>
        maxlen = 512
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.get_vocab()['</s>'])
        ONES = torch.ones_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([query_mask, ONES, doc_mask, ONES], dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)
        result = self.mt5.generate(input_ids=toks, attention_mask=mask, output_scores=True,
                                   return_dict_in_generate=True, max_length=2)
        cls_output = result.scores[0]
        cls_result = []
        for i in range(cls_output.shape[0] // BATCH):
            cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
        cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
        return cls_result


class MT5ClassificationHead(torch.nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim: int,
            inner_dim: int,
            num_classes: int,
            pooler_dropout: float,
    ):
        super().__init__()
        self.dense = torch.nn.Linear(input_dim, inner_dim)
        self.dropout = torch.nn.Dropout(p=pooler_dropout)
        self.out_proj = torch.nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class MT5ClassificationRanker(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.config = transformers.MT5Config.from_pretrained("google/mt5-base")
        self.mt5 = transformers.MT5Model.from_pretrained("google/mt5-base")
        self.tokenizer = transformers.T5Tokenizer.from_pretrained("google/mt5-base")
        self.classification_head = MT5ClassificationHead(self.config.d_model, self.config.d_model, 1, 0.1)
        self.vocab = self.tokenizer.get_vocab()

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        return toks

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 2  # 2x</s>
        maxlen = 512
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.eos_token_id)
        ONES = torch.ones_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([query_mask, ONES, doc_mask, ONES], dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        decoder_input_ids = shift_tokens_right(toks, self.config.pad_token_id, self.config.decoder_start_token_id)
        # execute BERT model
        outputs = self.mt5(input_ids=toks, attention_mask=mask, decoder_input_ids=decoder_input_ids)
        hidden_states = outputs[0]  # last hidden state

        eos_mask = toks.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                                  :, -1, :]
        logits = self.classification_head(sentence_representation)
        cls_output = logits
        cls_result = []
        for i in range(cls_output.shape[0] // BATCH):
            cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
        cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
        return cls_result


class MBARTRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bart_config = transformers.MBartConfig.from_pretrained("facebook/mbart-large-cc25")
        self.mbart = transformers.MBartForSequenceClassification.from_pretrained("facebook/mbart-large-cc25",
                                                                                 num_labels=1,
                                                                                 classifier_dropout=0.1)
        self.tokenizer = transformers.MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
        self.vocab = self.tokenizer.get_vocab()

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        return toks

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 2  # 2x</s>
        maxlen = 512
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.eos_token_id)
        ONES = torch.ones_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([query_mask, ONES, doc_mask, ONES], dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        # execute BERT model
        result = self.mbart(input_ids=toks, attention_mask=mask)

        cls_output = result.logits
        cls_result = []
        for i in range(cls_output.shape[0] // BATCH):
            cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
        cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
        return cls_result


class VanillaBertRanker(BertRanker):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        return self.cls(self.dropout(cls_reps[-1]))


class CustomBertModel(transformers.BertModel):
    """
    Based on pytorch_pretrained_bert.BertModel, but also outputs un-contextualized embeddings.
    """

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        Based on pytorch_pretrained_bert.BertModel
        """
        from transformers import MT5Tokenizer
        embedding_output = self.embeddings(input_ids, token_type_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(embedding_output, extended_attention_mask, return_dict=True,
                                      output_hidden_states=True)

        return [embedding_output] + list(encoded_layers.hidden_states)
