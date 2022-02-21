import torch
import torch.nn as nn
from data.data import get_iter
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


class LightningHyperparameters:
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

class BaseModel(LightningModule):
    def __init__(self, hparams: LightningHyperparameters):
        super(BaseModel, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """
        
        self.hparams = hparams
        self.label_size = hparams.bert_params['label_size']
        self.batch_size = hparams.bert_params['batch_size']
        self.loss_func = F.cross_entropy

    def forward(self, x, t, y, input_mask=None):
        pass

    def _initialize_weights(self):
        pass

    def configure_optimizers(self):
        parameters_list = self.get_trainable_params()[0]
        if parameters_list:
            return torch.optim.Adam(parameters_list, lr=0.0001)
        else:
            return [] # PyTorch Lightning hack for test mode with frozen model

    def get_trainable_params(self, recurse: bool = True) -> (List[nn.Parameter], int):
        parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        num_trainable_parameters = sum([p.flatten().size(0) for p in parameters])
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
        return {"loss": loss, "progress_bar": {"val_loss": loss, "val_accuracy": correct.mean(), "ite": torch.mean(torch.norm(diff, dim=1))},
                "log": {"batch_num": batch_idx, "val_loss": loss, "val_accuracy": correct.mean(), }, "correct": correct, "ite": torch.mean(torch.norm(diff, dim=1))}

    def validation_end(self, step_outputs):
        total_loss, total_correct, ites = list(), list(), list()
        for x in step_outputs:
            total_loss.append(x["loss"])
            total_correct.append(x["correct"])
            ites.append(x['ite'])
        avg_loss = torch.stack(total_loss).mean()
        accuracy = torch.cat(total_correct).mean()
        ate = torch.stack(ites).double().mean()
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


class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out # bs, num_of_basis


class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
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
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d

        x_treat_basis = self.spb.forward(x_treat) # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out


def comp_grid(y, num_grid):

    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int

    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L += (L < 0).int()

    return L.int().tolist(), U.int().tolist(), inter


class Density_Block(nn.Module):
    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
        self.ind = ind
        self.num_grid = num_grid
        self.outd = num_grid + 1

        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, t, x):
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        out = self.softmax(out)

        x1 = list(torch.arange(0, x.shape[0]))
        L, U, inter = comp_grid(t, self.num_grid)

        L_out = out[x1, L]
        U_out = out[x1, U]

        out = L_out + (U_out - L_out) * inter

        return out
        
# Targeted Regularizer

class TR(nn.Module):
    def __init__(self, degree, knots):
        super(TR, self).__init__()
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis
        self.weight = nn.Parameter(torch.rand(self.d), requires_grad=True)

    def forward(self, t):
        out = self.spb.forward(t)
        out = torch.matmul(out, self.weight)
        return out

    def _initialize_weights(self):
        #self.weight.data.normal_(0, 0.01)
        self.weight.data.zero_()

# ------------------------------------------ Drnet and Tarnet ------------------------------------------- #

class Treat_Linear(nn.Module):
    def __init__(self, ind, outd, act='relu', istreat=1, isbias=1, islastlayer=0):
        super(Treat_Linear, self).__init__()
        # ind does NOT include the extra concat treatment
        self.ind = ind
        self.outd = outd
        self.isbias = isbias
        self.istreat = istreat
        self.islastlayer = islastlayer

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        if self.istreat:
            self.treat_weight = nn.Parameter(torch.rand(1, self.outd), requires_grad=True)
        else:
            self.treat_weight = None

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
        x_feature = x[:, 1:]
        x_treat = x[:, [0]]

        out = torch.matmul(x_feature, self.weight)

        if self.istreat:
            out = out + torch.matmul(x_treat, self.treat_weight)
        if self.isbias:
            out = out + self.bias

        if self.act is not None:
            out = self.act(out)

        if not self.islastlayer:
            out = torch.cat((x_treat, out), 1)

        return out

class Multi_head(nn.Module):
    def __init__(self, cfg, isenhance):
        super(Multi_head, self).__init__()

        self.cfg = cfg # cfg does NOT include the extra dimension of concat treatment
        self.isenhance = isenhance  # set 1 to concat treatment every layer/ 0: only concat on first layer

        # we default set num of heads = 5
        self.pt = [0.0, 0.2, 0.4, 0.6, 0.8, 1.]

        self.outdim = -1
        # construct the predicting networks
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                self.outdim = layer_cfg[1]
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                           islastlayer=0))
        blocks.append(last_layer)
        self.Q1 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q2 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q3 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q4 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q5 = nn.Sequential(*blocks)

    def forward(self, x):
        # x = [treatment, features]
        out = torch.zeros(x.shape[0], self.outdim)
        t = x[:, 0]

        idx1 = list(set(list(torch.where(t >= self.pt[0])[0].numpy())) & set(torch.where(t < self.pt[1])[0].numpy()))
        idx2 = list(set(list(torch.where(t >= self.pt[1])[0].numpy())) & set(torch.where(t < self.pt[2])[0].numpy()))
        idx3 = list(set(list(torch.where(t >= self.pt[2])[0].numpy())) & set(torch.where(t < self.pt[3])[0].numpy()))
        idx4 = list(set(list(torch.where(t >= self.pt[3])[0].numpy())) & set(torch.where(t < self.pt[4])[0].numpy()))
        idx5 = list(set(list(torch.where(t >= self.pt[4])[0].numpy())) & set(torch.where(t <= self.pt[5])[0].numpy()))

        if idx1:
            out1 = self.Q1(x[idx1, :])
            out[idx1, :] = out[idx1, :] + out1

        if idx2:
            out2 = self.Q2(x[idx2, :])
            out[idx2, :] = out[idx2, :] + out2

        if idx3:
            out3 = self.Q3(x[idx3, :])
            out[idx3, :] = out[idx3, :] + out3

        if idx4:
            out4 = self.Q4(x[idx4, :])
            out[idx4, :] = out[idx4, :] + out4

        if idx5:
            out5 = self.Q5(x[idx5, :])
            out[idx5, :] = out[idx5, :] + out5

        return out


class DrnetAtt(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, isenhance):
        super(DrnetAtt, self).__init__()

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]

        self.cfg_density = cfg_density
        self.num_grid = num_grid
        self.cfg = cfg
        self.isenhance = isenhance

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*density_blocks)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(self.num_grid, density_hidden_dim, isbias=1)

        # multi-head outputs blocks
        self.Q = Multi_head(cfg, isenhance)

    def forward(self, t, x):
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
        g = self.density_estimator_head(t, hidden)
        Q = self.Q(t_hidden)

        return g, Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Treat_Linear):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()
                if m.istreat:
                    m.treat_weight.data.normal_(0, 1.)  # this needs to be initialized large to have better performance
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()

"""
cfg_density = [(3,4,1,'relu'), (4,6,1,'relu')]
cfg = [(6,4,1,'relu'), (4,1,1,'id')]
num_grid = 10
isenhance = 1
D = Drnet(cfg_density, num_grid, cfg, isenhance)
D._initialize_weights()
x = torch.rand(10, 3)
t = torch.rand(10)
y = torch.rand(10)
out = D.forward(t, x)
"""

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransIPT(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, isenhance):
        super(TransIPT, self).__init__()

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]

        self.cfg_density = cfg_density
        self.num_grid = num_grid
        self.cfg = cfg
        self.isenhance = isenhance

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        encoder_layers = TransformerEncoderLayer(d_model=50, nhead=2, d_hid=50, dropout=0.1)
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(TransformerEncoder(encoder_layers, nlayers=1))
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*density_blocks)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(self.num_grid, density_hidden_dim, isbias=1)

        # multi-head outputs blocks
        self.Q = Multi_head(cfg, isenhance)

    def forward(self, t, x):
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
        g = self.density_estimator_head(t, hidden)
        Q = self.Q(t_hidden)

        return g, Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Treat_Linear):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()
                if m.istreat:
                    m.treat_weight.data.normal_(0, 1.)  # this needs to be initialized large to have better performance
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()