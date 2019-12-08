snli:{
    dev:std.extVar('PWD') + '/data/snli_1.0/multinli_1.0_dev_matched.jsonl',
    train:std.extVar('PWD') + '/data/snli_1.0/multinli_1.0_train.jsonl',
    test:std.extVar('PWD') + '/data/snli_1.0/snli_1.0_test.jsonl',
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
    train:std.extVar('PWD') + '/data/cache/train',
    dev:std.extVar('PWD') + '/data/cache/dev',
    test:std.extVar('PWD') + '/data/cache/test',
},
models:{
    allen_ner:std.extVar('PWD') + '/data/models/ner-model-2018.12.18.tar.gz',
}