from allennlp.data.dataset_readers import Conll2003DatasetReader

train_dataset = Conll2003DatasetReader('./data/conll/train.txt')
test_dataset = Conll2003DatasetReader('./data/conll/test.txt')
