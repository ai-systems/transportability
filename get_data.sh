mkdir -p data
cd data
wget -nc 'http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.1.zip'
unzip -o 'SciTailV1.1.zip'
wget -nc 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
unzip -o 'snli_1.0.zip'
wget -nc 'https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip'
unzip -o 'multinli_1.0.zip'
mkdir -p  ner
cd ner
mkdir -p conll2003
wget -nc 'https://github.com/synalp/NER/raw/master/corpus/CoNLL-2003/eng.train' -P conll2003
wget -nc 'https://github.com/synalp/NER/raw/master/corpus/CoNLL-2003/eng.testa' -P conll2003
wget -nc 'https://github.com/synalp/NER/raw/master/corpus/CoNLL-2003/eng.testb' -P conll2003
mkdir -p wiki
wget -nc 'https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/wikigold/CONLL-format/data/wikigold.conll.txt' -P wiki
mkdir -p twitter
wget -nc 'https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/WNUT17/CONLL-format/data/test/emerging.test.annotated' -P twitter
wget -nc 'https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/WNUT17/CONLL-format/data/dev/emerging.dev.conll' -P twitter
wget -nc 'https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/WNUT17/CONLL-format/data/train/wnut17train.conll' -P twitter
cd ..
mkdir models
wget -nc 'https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz' -P models

