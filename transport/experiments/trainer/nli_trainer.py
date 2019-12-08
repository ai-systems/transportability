import os
import random
from functools import reduce

import luigi
import numpy as np
import torch
from overrides import overrides
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (AdamW, BertConfig, BertForSequenceClassification,
                          get_linear_schedule_with_warmup)

from regra.experiments.abc.config_experiment import ConfigExperiment
from transport.datasets.snli import SnliDataset

TEST = 'test'
TRAIN = 'train'
DEV = 'dev'

BERT_MODEL = 'bert-base-uncased'


EXPERIMENT_MPK = 'nli_experiment.mpk'


class NLIExperiment(ConfigExperiment):
    mode = luigi.Parameter(default='train')
    @overrides
    def requires(self):
        return [SnliDataset(config_file=self.config_file, mode=TEST),
                SnliDataset(config_file=self.config_file, mode=DEV),
                SnliDataset(config_file=self.config_file, mode=TRAIN)
                ]

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(seed)

    @overrides
    def run(self):
        num_train_epochs = 3
        gradient_accumulation_steps = 1
        weight_decay = 0.0
        learning_rate = 5e-5
        adam_epsilon = 1e-8
        warmup_steps = 0
        seed = 42
        logging_steps = 50

        if torch.cuda.device_count() == 1:
            model = model.cuda()
        elif torch.cuda.device_count() > 1:
            model = model.cuda()
            model = torch.nn.DataParallel(model)

        train_batch_size = 8 * max(1, torch.cuda.device_count())
        train_dataloader = DataLoader(SnliDataset(config_file=self.config_file, mode=TRAIN).get_dataset(
        ), batch_size=train_batch_size, shuffle=True)
        dev_loader = DataLoader(SnliDataset(config_file=self.config_file, mode=DEV).get_dataset(
        ), batch_size=train_batch_size, shuffle=True)
        test_loader = DataLoader(SnliDataset(config_file=self.config_file, mode=DEV).get_dataset(
        ), batch_size=train_batch_size, shuffle=True)
        t_total = len(
            train_dataloader) // gradient_accumulation_steps * num_train_epochs
        if self.mode == 'train':
            self.logger.info('Loading pretrained model')
            config = BertConfig.from_pretrained(BERT_MODEL, num_labels=3)
            model = BertForSequenceClassification.from_pretrained(
                BERT_MODEL, config=config)
        else:
            self.logger.info('Loading trained model from local directory')
            config = BertConfig.from_json_file(
                f'{self.path}/checkpoint-best/config.json')
            model = BertForSequenceClassification.from_pretrained(
                f'{self.path}/checkpoint-best/pytorch_model.bin', config=config)

        if torch.cuda.device_count() == 1:
            model = model.cuda()
            self.logger.info('GPUs used: 1')
        elif torch.cuda.device_count() > 1:
            model = model.cuda()
            model = torch.nn.DataParallel(model)
            self.logger.info(f'GPUs used: {torch.cuda.device_count()}')
        else:
            self.logger.warn('No GPUs used!')

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        global_step, accuracy = 0, 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        # Added here for reproductibility (even between python 2 and 3)
        self.set_seed(seed)
        if self.mode == 'train':
            self.logger.info('Running training')
            for _ in train_iterator:
                epoch_iterator = tqdm(train_dataloader, desc="Train Iteration")
                for step, batch in enumerate(epoch_iterator):
                    model.train()
                    batch = tuple(t.cuda() for t in batch)
                    inputs = {'input_ids':      batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2],
                              'labels':         batch[3]}
                    outputs = model(**inputs)
                    # model outputs are always tuple in transformers (see doc)
                    loss = outputs[0]

                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps

                    loss.backward()

                    tr_loss += loss.item()
                    if (step + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1
                        if global_step % logging_steps == 0:
                            epoch_iterator.set_description(
                                f'Loss: {(tr_loss - logging_loss)/logging_steps}')
                            logging_loss = tr_loss
                eval_acc = self.evaluate(dev_loader, model)
                self.logger.info(f'Dev accuracy: {eval_acc}')
                if accuracy < eval_acc:
                    output_dir = os.path.join(
                        self.path, 'checkpoint-{}'.format('best'))
                    self.logger.info(f'Saving best model to {output_dir}')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        else:
            eval_acc = self.evaluate(dev_loader, model)
            self.logger.info(f'Dev Accuracy: {eval_acc}')
            eval_acc = self.evaluate(test_loader, model)
            self.logger.info(f'Test Accuracy: {eval_acc}')

    def evaluate(self, dataloader, model):
        self.logger.info('Running evalution of model')
        preds = None
        with torch.no_grad():
            epoch_iterator = tqdm(dataloader, desc="Test Iteration")
            for step, batch in enumerate(epoch_iterator):
                model.eval()
                batch = tuple(t.cuda() for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels':         batch[3]}
                outputs = model(**inputs)
                _, logits = outputs[:2]
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(
                        preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        preds = np.argmax(preds, axis=1)
        accuracy = (preds == out_label_ids).mean()
        return accuracy

    @overrides
    def output(self):
        return luigi.LocalTarget(path=self.cache_path)

    @property
    def cache_path(self):
        return f'{self.path}/{EXPERIMENT_MPK}'
