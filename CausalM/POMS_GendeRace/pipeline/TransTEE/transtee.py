import torch
import torch.nn as nn
from TransTEE.transformers import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from TransTEE.trans_ci import TransformerModel, Embeddings
from TransTEE.dynamic_net import Density_Block, Dynamic_FC, Truncated_power, Treat_Linear
import torch.nn.functional as F
from pytorch_lightning import LightningModule, data_loader
from BERT.bert_text_dataset import BERT_PRETRAINED_MODEL, BertTextDataset, InputExample, InputLabel, InputFeatures, \
    truncate_seq_first
from constants import NUM_CPU, MAX_SENTIMENT_SEQ_LENGTH
from datasets.utils import CLS_TOKEN, SEP_TOKEN
from typing import Callable, List
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from utils import save_predictions
from transformers import BertModel
# replace the feature extractor of x by self-attention
# 0.015
class Linear(nn.Module):
    def __init__(self, ind, outd, act='relu', isbias=1):
        super(Linear, self).__init__()
        # ind does NOT include the extra concat treatment
        self.ind = ind
        self.outd = outd
        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)

        out = torch.matmul(x, self.weight)

        if self.isbias:
            out = out + self.bias

        if self.act is not None:
            out = self.act(out)

        return out

class LightningHyperparameters:
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

class TransTEE(LightningModule):
    def __init__(self, hparams: LightningHyperparameters, embed_size = 768, num_t=1,  num_heads=4, att_layers=1, dropout=0.1, init_range_f=0.1, init_range_t=0.2):
        super(TransTEE, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """
        self.att_layers = att_layers
        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        
        self.hparams = hparams
        num_cov = hparams.max_seq_len
        self.label_size = hparams.bert_params['label_size']
        self.batch_size = hparams.bert_params['batch_size']
        
        self.feature_weight = Embeddings(embed_size, initrange=init_range_f)
        self.treat_emb = Embeddings(embed_size, act='id', initrange=init_range_t)

        encoder_layers = TransformerEncoderLayer(embed_size, nhead=num_heads, dropout=dropout, num_cov=num_cov)
        self.encoder = TransformerEncoder(encoder_layers, att_layers)

        decoder_layers = TransformerDecoderLayer(embed_size, nhead=num_heads, dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layers, att_layers)

        self.Q = nn.Linear(embed_size, self.label_size)
        self.loss_func = F.cross_entropy

        self.bert = TransTEE.load_frozen_bert()
        self.hidden_size = self.bert.config.hidden_size
        # self.pooler = HAN_Attention_Pooler_Layer(self.hidden_size)
        # self.classifier = Linear_Layer(self.hidden_size, self.label_size, dropout, activation=None)

    def forward(self, x, t, y, input_mask=None):
        memory, _ = self.bert(x, attention_mask=input_mask)
        #hidden = self.feature_weight(x)
        # 
        # memory = self.encoder(hidden, src_key_padding_mask=input_mask)

        tgt = self.treat_emb(t)
        if len(tgt.shape) < 3:
            tgt = tgt.unsqueeze(1)
        input_mask = ~input_mask.bool() if input_mask is not None else None
        out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2), memory_key_padding_mask=input_mask)

        logits = self.Q(out.squeeze(0))
        loss = self.loss_func(logits.view(-1, self.label_size), y.view(-1))
        return loss, logits, None

    @staticmethod
    def load_frozen_bert(bert_pretrained_model: str = 'bert-base-cased', bert_state_dict: str = None) -> BertModel:
        if bert_state_dict:
            fine_tuned_state_dict = torch.load(bert_state_dict)
            bert = BertModel.from_pretrained(bert_pretrained_model, state_dict=fine_tuned_state_dict)
        else:
            bert = BertModel.from_pretrained(bert_pretrained_model)
        for p in bert.parameters():
            p.requires_grad = False
        return bert

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.in_features == 1:
                    continue
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def configure_optimizers(self):
        parameters_list = self.get_trainable_params()[0]
        if parameters_list:
            return torch.optim.Adam(parameters_list, lr=0.001)
        else:
            return [] # PyTorch Lightning hack for test mode with frozen model

    def get_trainable_params(self, recurse: bool = True) -> (List[nn.Parameter], int):
        parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        num_trainable_parameters = sum([p.flatten().size(0) for p in parameters])
        print('model_params', num_trainable_parameters)
        return parameters, num_trainable_parameters

    @data_loader
    def train_dataloader(self):
        if not self.training:
            return [] # PyTorch Lightning hack for test mode with frozen model
        dataset = BertTextClassificationDataset(self.hparams.data_path, self.hparams.treatment, "train",
                                                self.hparams.text_column, self.hparams.label_column,
                                                max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def training_step(self, batch, batch_idx):
        input_ids, cf_unique_ids, input_mask, labels, unique_ids,treatment_labels = batch
        loss, logits, attention_weight = self.forward(input_ids, treatment_labels, labels, input_mask=input_mask)
        predictions = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()
        total = torch.tensor(predictions.size(0))
        return {"loss": loss, "log": {"batch_num": batch_idx, "train_loss": loss, "train_accuracy": correct.mean()},
                "correct": correct.sum(), "total": total}

    @data_loader
    def val_dataloader(self):
        if not self.training:
            return [] # PyTorch Lightning hack for test mode with frozen model
        dataset = BertTextClassificationDataset(self.hparams.data_path, self.hparams.treatment, "dev",
                                                self.hparams.text_column, self.hparams.label_column,
                                                max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def validation_step(self, batch, batch_idx):
        input_ids, cf_unique_ids, input_mask, labels, unique_ids,treatment_labels = batch
        loss, logits, pooler_attention_weights = self.forward(input_ids, treatment_labels, labels, input_mask=input_mask)
        prediction_probs = F.softmax(logits.view(-1, self.label_size), dim=-1)
        predictions = torch.argmax(prediction_probs, dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()

        loss, logits, pooler_attention_weights = self.forward(input_ids, 1-treatment_labels, labels, input_mask=input_mask)
        prediction_probs_cf = F.softmax(logits.view(-1, self.label_size), dim=-1)
        diff = torch.abs(prediction_probs_cf-prediction_probs)

        inds_max = torch.norm(diff, dim=1).topk(5, dim=0, largest=True, sorted=True)[1]
        inds_min = torch.norm(diff, dim=1).topk(5, dim=0, largest=False, sorted=True)[1]
        return {"loss": loss, "progress_bar": {"val_loss": loss, "val_accuracy": correct.mean(), "ite": torch.mean(torch.norm(diff, dim=1))},
                "log": {"batch_num": batch_idx, "val_loss": loss, "val_accuracy": correct.mean(), }, "correct": correct, "ite": torch.norm(diff, dim=1), 'inds_max': inds_max, "inds_min": inds_min, "unique_ids": unique_ids}

    def validation_end(self, step_outputs):
        total_loss, total_correct, ites, ids = list(), list(), list(), list()
        for x in step_outputs:
            total_loss.append(x["loss"])
            total_correct.append(x["correct"])
            ites.append(x['ite'])
            ids.append(x['unique_ids'])
        avg_loss = torch.stack(total_loss).mean()
        accuracy = torch.cat(total_correct).mean()
        ate = torch.stack(ites[:-1]).double().mean()

        ate_all = torch.cat(ites, dim=0)
        indexs = torch.cat(ids, dim=0)
        ites_max = ate_all.topk(10, dim=0, largest=True, sorted=True)
        ites_min = ate_all.topk(10, dim=0, largest=False, sorted=True)
        print('\n max ate samples', indexs[ites_max[1]])
        print("max ites", ites_max[0])
        print('min ate samples', indexs[ites_min[1]])
        print("min ites", ites_min[0])
        return {"loss": avg_loss, "progress_bar": {"val_loss": avg_loss, "val_accuracy": accuracy , "ate": ate},
                "log": {"val_loss": avg_loss, "val_accuracy": accuracy}}

    @data_loader
    def test_dataloader(self):
        dataset = BertTextClassificationDataset(self.hparams.data_path, self.hparams.treatment, "test",
                                                self.hparams.text_column, self.hparams.label_column,
                                                max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def test_step(self, batch, batch_idx):
        input_ids, cf_input_ids, input_mask, labels, unique_ids,treatment_labels = batch
        loss, logits, pooler_attention_weights = self.forward(input_ids, treatment_labels, labels, input_mask=input_mask)
        prediction_probs = F.softmax(logits.view(-1, self.label_size), dim=-1)
        predictions = torch.argmax(prediction_probs, dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()

        loss, logits, pooler_attention_weights = self.forward(input_ids, 1-treatment_labels, labels, input_mask=input_mask)
        prediction_probs_cf = F.softmax(logits.view(-1, self.label_size), dim=-1)
        diff = torch.abs(prediction_probs_cf-prediction_probs)


        return {"loss": loss, "progress_bar": {"test_loss": loss, "test_accuracy": correct.mean()},
                "log": {"batch_num": batch_idx, "test_loss": loss, "test_accuracy": correct.mean()},
                "predictions": predictions, "labels": labels, "treatment_labels": treatment_labels, "unique_ids": unique_ids, "prediction_probs": prediction_probs, 'ite': torch.mean(torch.norm(diff, dim=1))}

    def test_end(self, step_outputs):
        total_loss, total_predictions, total_labels, total_unique_ids, total_prediction_probs, ites = list(), list(), list(), list(), list(), list()
        for x in step_outputs:
            total_loss.append(x["loss"])
            total_predictions.append(x["predictions"])
            total_labels.append(x["labels"])
            total_unique_ids.append(x["unique_ids"])
            total_prediction_probs.append(x["prediction_probs"])
            ites.append(x['ite'])
        avg_loss = torch.stack(total_loss).double().mean()
        unique_ids = torch.cat(total_unique_ids).long()
        predictions = torch.cat(total_predictions).long()
        ate = torch.stack(ites).double().mean()
        prediction_probs = torch.cat(total_prediction_probs, dim=0).double()
        labels = torch.cat(total_labels).long()
        correct = predictions.eq(labels.view_as(predictions)).long()
        accuracy = correct.double().mean()
        save_predictions(self.hparams.output_path,
                         unique_ids.data.cpu().numpy(),
                         predictions.data.cpu().numpy(),
                         labels.data.cpu().numpy(),
                         correct.cpu().numpy(),
                         [prediction_probs[:, i].data.cpu().numpy() for i in range(self.label_size)],
                         f"transtee-test")
        return {"loss": avg_loss, 
                "progress_bar": {"test_loss": avg_loss, "test_accuracy": accuracy, "ate": ate},
                "log": {"test_loss": avg_loss, "test_accuracy": accuracy}}


class BertTextClassificationDataset(BertTextDataset):

    def __init__(self, data_path: str, treatment: str, subset: str, text_column: str, label_column: str,
                 bert_pretrained_model: str = BERT_PRETRAINED_MODEL, max_seq_length: int = MAX_SENTIMENT_SEQ_LENGTH):
        super().__init__(data_path, treatment, subset, text_column, label_column, bert_pretrained_model, max_seq_length)

    def read_examples_func(self, row):
        return InputExample(unique_id=int(row.iloc[0]), text=str(row[self.text_column]), cf_text_colomn=str(row[self.cf_text_colomn]), label=int(row[self.label_column]), treatment_label=int(row[self.treatment_label]))

    def convert_examples_to_features(self, examples):
        """Loads a data file into a list of `InputFeature`s."""
        features_list = list()
        cf_features_list = list()
        labels_list = list()
        for i, example in tqdm(enumerate(examples), total=len(examples), desc=f"{self.subset}-convert_examples_to_features"):
            features, example_len = self.tokenize_and_pad_sequence(example)
            cf_features, example_len = self.tokenize_and_pad_sequence_cf(example)
            features_list.append(features)
            labels_list.append(InputLabel(unique_id=example.unique_id, label=example.label, treatment_label=example.treatment_label))
        return features_list, labels_list

    def tokenize_and_pad_sequence(self, example):
        tokens = self.tokenizer.tokenize(example.text)

        tokens = tuple([CLS_TOKEN] + truncate_seq_first(tokens, self.max_seq_length) + [SEP_TOKEN])

        example_len = len(tokens) - 2

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(self.PAD_TOKEN_IDX)
            input_mask.append(self.PAD_TOKEN_IDX)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length

        return InputFeatures(unique_id=example.unique_id, tokens=tokens,
                             input_ids=input_ids, input_mask=input_mask), example_len

    def convert_examples_to_features(self, examples):
        """Loads a data file into a list of `InputFeature`s."""
        features_list = list()
        cf_features_list = list()
        labels_list = list()
        for i, example in tqdm(enumerate(examples), total=len(examples), desc=f"{self.subset}-convert_examples_to_features"):
            features, example_len = self.tokenize_and_pad_sequence(example)
            cf_features, example_len = self.tokenize_and_pad_sequence_cf(example)
            features_list.append(features)
            cf_features_list.append(cf_features)
            labels_list.append(InputLabel(unique_id=example.unique_id, label=example.label, treatment_label=example.treatment_label))
        return features_list, cf_features_list, labels_list

    def tokenize_and_pad_sequence_cf(self, example):
        tokens = self.tokenizer.tokenize(example.cf_text_colomn)

        tokens = tuple([CLS_TOKEN] + truncate_seq_first(tokens, self.max_seq_length) + [SEP_TOKEN])

        example_len = len(tokens) - 2

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(self.PAD_TOKEN_IDX)
            input_mask.append(self.PAD_TOKEN_IDX)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length

        return InputFeatures(unique_id=example.unique_id, tokens=tokens,
                             input_ids=input_ids, input_mask=input_mask), example_len
