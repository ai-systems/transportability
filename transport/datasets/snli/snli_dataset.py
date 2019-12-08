import luigi
import ujson as json
from overrides import overrides
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset

from regra.dataset.abc import MsgPackDataset
from .torch import SNLITorchDataset

SNLI_DATASET = 'snli_dataset.mpk'

BERT_MODEL = 'bert-base-uncased'
MAX_LENGTH = 512

LABEL_MAP = {
    'neutral': 0,
    'contradiction': 1,
    'entailment': 2
}


class SnliDataset(MsgPackDataset):

    dataset_path = luigi.Parameter(default=None)

    @overrides
    def requires(self):
        return []

    @staticmethod
    def convert_to_features(sentence1, sentence2, tokenizer, pad_on_left=True, pad_token=0,
                            pad_token_segment_id=0,
                            mask_padding_with_zero=True):
        inputs = tokenizer.encode_plus(sentence1, sentence2, add_special_tokens=True,
                                       max_length=MAX_LENGTH)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = MAX_LENGTH - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1]
                              * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] *
                              padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + \
                ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + \
                ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == MAX_LENGTH, "Error with input length {} vs {}".format(
            len(input_ids), MAX_LENGTH)
        assert len(attention_mask) == MAX_LENGTH, "Error with input length {} vs {}".format(
            len(attention_mask), MAX_LENGTH)
        assert len(token_type_ids) == MAX_LENGTH, "Error with input length {} vs {}".format(
            len(token_type_ids), MAX_LENGTH)
        return input_ids, attention_mask, token_type_ids

    @overrides
    def run(self):
        self.logger.info(f'Dataset path {self.dataset_path}')
        bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        skipped = 0
        processed_data = {}
        count = 0
        with open(self.dataset_path, 'r') as jsonl_file:
            json_data = [json.loads(line) for line in jsonl_file]
            for data in tqdm(json_data, 'Preporcessing SNLI'):
                id, sentence1, sentence2, gold_label = count, data[
                    'sentence1'], data['sentence2'], data['gold_label']
                count += 1
                if gold_label not in LABEL_MAP:
                    self.logger.warn(
                        f'{id} doesnot have mapped gold labels. Label - {gold_label}. Skipping it')
                    skipped += 1
                    continue

                input_ids, attention_mask, token_type_ids = self.convert_to_features(
                    sentence1, sentence2, bert_tokenizer)
                label = LABEL_MAP[gold_label]
                processed_data[id] = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'label': label
                }
        self.output().dump(processed_data)
        self.logger.warn(f'Skipped: {skipped} out of {len(json_data)}')

    def get_dataset(self):
        data = self.output().load()
        return SNLITorchDataset(data)

    @property
    def cache_path(self):
        return f'{self.path}/{SNLI_DATASET}'
