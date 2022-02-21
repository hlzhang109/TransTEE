from random import random
from torch.utils.data import TensorDataset, Dataset
from transformers.tokenization_bert import BertTokenizer
from constants import BERT_PRETRAINED_MODEL, MAX_SENTIMENT_SEQ_LENGTH
from abc import abstractmethod
from typing import List
import pandas as pd
import numpy as np
import torch


class InputExample:
    def __init__(self, unique_id, text,  label, cf_text_colomn, treatment_label):
        self.unique_id = unique_id
        self.text = text
        self.cf_text_colomn = cf_text_colomn
        self.label = label
        self.treatment_label = treatment_label


class InputFeatures:
    def __init__(self, unique_id, tokens, input_ids, input_mask):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask


class InputLabel:
    def __init__(self, unique_id, label, treatment_label):
        self.unique_id = unique_id
        self.label = label
        self.treatment_label = treatment_label


class BertTextDataset(Dataset):

    PAD_TOKEN_IDX = 0
    MLM_IGNORE_LABEL_IDX = -1

    def __init__(self, data_path: str, treatment: str, subset: str, text_column: str, label_column: str,
                 bert_pretrained_model: str = BERT_PRETRAINED_MODEL, max_seq_length: int = MAX_SENTIMENT_SEQ_LENGTH):
        super().__init__()
        if subset not in ("train", "dev", "test", "train_debug", "dev_debug", "test_debug"):
            raise ValueError("subset argument must be {train, dev,test}")
        self.dataset_file = f"{data_path}/{treatment}_{subset}.csv"
        t = treatment.capitalize()
        self.treatment_label = f"{t}_F_label"
        self.subset = subset
        self.text_column = text_column
        if 'CF' in text_column:
            self.cf_text_colomn = text_column[:-2] + 'F'
        else:
            self.cf_text_colomn = text_column[:-1] + 'CF'
        self.label_column = label_column
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model,
                                                       do_lower_case=bool(BERT_PRETRAINED_MODEL.endswith("uncased")))
        self.dataset = self.preprocessing_pipeline()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def preprocessing_pipeline(self):
        examples = self.read_examples()
        features, cf_features, labels = self.convert_examples_to_features(examples)
        dataset = self.create_tensor_dataset(features, cf_features=cf_features, labels=labels)
        return dataset

    def read_examples(self) -> List[InputExample]:
        """Read a list of `InputExample`s from an input file."""
        df = pd.read_csv(self.dataset_file, header=0, encoding='utf-8')
        return df.apply(self.read_examples_func, axis=1).tolist()

    @abstractmethod
    def read_examples_func(self, row: pd.Series) -> InputExample: ...

    @abstractmethod
    def convert_examples_to_features(self, examples: List[InputExample]) -> (List[InputFeatures], List[InputLabel]): ...

    @staticmethod
    def create_tensor_dataset(features: List[InputFeatures], cf_features: List[InputFeatures], labels: List[InputLabel]) -> TensorDataset:
        input_ids_list = list()
        input_cf_ids_list = list()
        input_masks_list = list()
        input_unique_id_list = list()
        input_labels_list = list()
        input_treatment_labels_list = list()
        for f, cf, l in zip(features, cf_features, labels):
            input_ids_list.append(f.input_ids)
            input_cf_ids_list.append(cf.input_ids)
            input_masks_list.append(f.input_mask)
            assert l.unique_id == f.unique_id
            input_unique_id_list.append(f.unique_id)
            input_labels_list.append(l.label)
            input_treatment_labels_list.append(l.treatment_label)
        all_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        all_input_ids_cf = torch.tensor(input_cf_ids_list, dtype=torch.long)
        all_input_mask = torch.tensor(input_masks_list, dtype=torch.long)
        all_labels = torch.tensor(input_labels_list, dtype=torch.long)
        all_treatment_labels = torch.tensor(input_treatment_labels_list, dtype=torch.long)
        all_unique_id = torch.tensor(input_unique_id_list, dtype=torch.long)

        return TensorDataset(all_input_ids, all_input_ids_cf, all_input_mask, all_labels, all_unique_id, all_treatment_labels)

def print_seq_lengths_stats(logger, text_seq_lengths, max_seq_length):
    logger.info(f"Num Sequences: {len(text_seq_lengths)}")
    logger.info(f"Minimum Sequence Length: {np.min(text_seq_lengths)}")
    logger.info(f"Average Sequence Length: {np.mean(text_seq_lengths)}")
    logger.info(f"Median Sequence Length: {np.median(text_seq_lengths)}")
    logger.info(f"99th Percentile Sequence Length: {np.percentile(text_seq_lengths, 99)}")
    logger.info(f"Maximum Sequence Length: {np.max(text_seq_lengths)}")
    logger.info(f"Num of over Maximum Sequence Length: {len([i for i in text_seq_lengths if i >= max_seq_length])}")


def truncate_seq_random_sub(tokens, max_seq_length):
    max_num_tokens = max_seq_length - 2
    l = 0
    r = len(tokens)
    trunc_tokens = list(tokens)
    while r - l > max_num_tokens:
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            l += 1
        else:
            r -= 1
    return trunc_tokens[l:r]


def truncate_seq_first(tokens, max_seq_length):
    max_num_tokens = max_seq_length - 2
    trunc_tokens = list(tokens)
    if len(trunc_tokens) > max_num_tokens:
        trunc_tokens = trunc_tokens[:max_num_tokens]
    return trunc_tokens