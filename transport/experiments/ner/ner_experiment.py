from functools import reduce
from .connlleval import evaluate
from regra.experiments.abc.config_experiment import ConfigExperiment
from collections import defaultdict

import spacy
from luigi import LocalTarget, Parameter
from overrides import overrides
from spacy.tokenizer import Tokenizer
from tqdm import tqdm
from nltk.tag import StanfordNERTagger
st = StanfordNERTagger('./data/models/stanford-ner-2018-10-16/classifiers/english.conll.4class.distsim.crf.ser.gz',
                       './data/models/stanford-ner-2018-10-16/stanford-ner.jar', encoding='utf-8')


nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = Tokenizer(nlp.vocab)

SPACY_CONLL_MAPPING = defaultdict(lambda: 'MISC', {
    'PERSON': 'PER',
    'ORG': 'ORG',
    'LOC': 'LOC'
})

STANFORD_CONLL_MAPPING = defaultdict(lambda: 'MISC', {
    'PERSON': 'PER',
    'ORGANIZATION': 'ORG',
    'LOCATION': 'LOC'
})

SPACY_WIKI_MAPPING = defaultdict(lambda: 'MISC', {
    'PERSON': 'PER',
    'ORG': 'ORG',
    'LOC': 'LOC'
})

SPACY_WNUT_MAPPING = defaultdict(lambda: 'MISC', {
    'PERSON': 'person',
    'LOC': 'location'
})

STANFORD_WNUT_MAPPING = defaultdict(lambda: 'MISC', {
    'PERSON': 'person',
    'LOCATION': 'location'
})


class NerExperiment(ConfigExperiment):

    ner_dataset_path = Parameter(default=None)
    pre_trained = Parameter(default=None)
    dataset = Parameter(default=None)
    accepted_tags = Parameter(default=None)
    allen_ner_model = Parameter(default=None)

    @overrides
    def requires(self):
        return []

    @overrides
    def output(self):
        return LocalTarget('./data/temp.mpk')

    def convert_ent(self, type):
        if self.dataset == 'conll' and self.pre_trained == 'spacy':
            return SPACY_CONLL_MAPPING[type]
        if self.dataset == 'conll' and self.pre_trained == 'stanford':
            return STANFORD_CONLL_MAPPING[type]
        if self.dataset == 'wiki' and self.pre_trained == 'stanford':
            return STANFORD_CONLL_MAPPING[type]
        if self.dataset == 'wiki' and self.pre_trained == 'stanford':
            return STANFORD_WNUT_MAPPING[type]

        else:
            return SPACY_WIKI_MAPPING[type]

    @staticmethod
    def to_conll_iob(annotated_sentence):
        proper_iob_tokens = []
        for idx, annotated_token in enumerate(annotated_sentence):
            word, ner = annotated_token

            if ner != 'O':
                if idx == 0:
                    ner = "B-" + ner
                elif annotated_sentence[idx - 1][1] == ner:
                    ner = "I-" + ner
                else:
                    ner = "B-" + ner
            proper_iob_tokens.append((word, ner))
        return proper_iob_tokens

    def predict(self, words_or_sentences, predictor=None):
        tags, extracted_tokens = [], []
        if self.pre_trained == 'spacy':
            sentence = ' '.join(words_or_sentences)
            docs = nlp(sentence)
            extracted_tokens = [doc.text for doc in docs]
            tags = [f'{doc.ent_iob_}-{self.convert_ent(doc.ent_type_)}' if doc.ent_iob_ !=
                    'O' else 'O' for doc in docs]
            # print(tags)
        elif self.pre_trained == 'stanford':
            tagged_sentences = st.tag_sents(words_or_sentences)
            tags = [f'{tag.split("-")[0]}-{self.convert_ent(tag.split("-")[1])}' if tag != 'O' else 'O' for tagged_words in tagged_sentences for _,
                    tag in self.to_conll_iob(tagged_words)]
        if len(tags) != len(reduce(list.__add__, words_or_sentences)):
            self.logger.warn(f'There is a tag mismatch for {" ".join(words)}')
        return tags

    @overrides
    def run(self):
        ner_file = []
        with open(self.ner_dataset_path) as f:
            for line in f:
                ner_file.append(line)

        words = []
        true_tags, pred_tags = [], []
        sentences = []
        for line in tqdm(ner_file, desc='Processing NER file'):
            line = line.strip()
            if '-DOCSTART' in line:
                continue
            elif line == '':
                if len(words) > 0:
                    if self.pre_trained == 'stanford':
                        sentences.append(words)
                        if len(sentences) > 100:
                            predicted_tags = self.predict(sentences)
                            predicted_tags = ['O' if tag != 'O' and tag.split(
                                '-')[1] not in self.accepted_tags else tag for tag in predicted_tags]
                            sentences = []
                            pred_tags.extend(predicted_tags)
                    else:
                        predicted_tags = self.predict(words)
                        predicted_tags = ['O' if tag != 'O' and tag.split(
                            '-')[1] not in self.accepted_tags else tag for tag in predicted_tags]
                        pred_tags.extend(predicted_tags)
                    words = []
            else:
                if self.dataset == 'wiki' or self.dataset == 'twitter':
                    word,  tag = line.split()
                else:
                    word, _, _, tag = line.split()
                words.append(word)
                if tag != 'O' and tag.split('-')[1] not in self.accepted_tags:
                    true_tags.append('O')
                else:
                    true_tags.append(tag)
        evaluate(true_tags, pred_tags)
