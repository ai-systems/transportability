snli:{
    dev:std.extVar('PWD') + '/data/SciTailV1.1/snli_format/scitail_1.0_dev.txt',
    train:std.extVar('PWD') + '/data/SciTailV1.1/snli_format/scitail_1.0_train.txt',
    test:std.extVar('PWD') + '/data/SciTailV1.1/snli_format/scitail_1.0_test.txt',
},
ner:{
    conll:{
        dev:std.extVar('PWD') + '/data/ner/conll2003/eng.testa',
        train:std.extVar('PWD') + '/data/ner/conll2003/eng.train',
        test:std.extVar('PWD') + '/data/ner/conll2003/eng.testb',
    },
    wiki: std.extVar('PWD') + '/data/ner/wiki/wikigold.conll.txt',
    twitter:{
        dev:std.extVar('PWD') + '/data/ner/twitter/emerging.dev.annotated',
        train:std.extVar('PWD') + '/data/ner/twitter/wnut17train.conll',
        test:std.extVar('PWD') + '/data/ner/twitter/emerging.test.annotated',
    }
},
cache_path:{
    model:std.extVar('PWD') + '/data/cache/train_multi',
    train:std.extVar('PWD') + '/data/cache/train_multi',
    dev:std.extVar('PWD') + '/data/cache/dev_multi',
    test:std.extVar('PWD') + '/data/cache/test_multi',
},

models:{
    allen_ner:std.extVar('PWD') + '/data/models/ner-model-2018.12.18.tar.gz',
}