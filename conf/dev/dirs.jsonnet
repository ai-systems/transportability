snli:{
    dev:std.extVar('PWD') + '/tests/resources/snli/snli.jsonl',
    train:std.extVar('PWD') + '/tests/resources/snli/snli.jsonl',
    test:std.extVar('PWD') + '/tests/resources/snli/snli.jsonl',
}, 
ner:{
    conll:{
        dev:std.extVar('PWD') + '/tests/resources/eng.testa',
        train:std.extVar('PWD') + '/tests/resources/eng.testa',
        test:std.extVar('PWD') + '/tests/resources/eng.testa',
    },
},
cache_path:{
    train:std.extVar('PWD') + '/tests/resources/cache',
    dev:std.extVar('PWD') + '/tests/resources/cache',
    test:std.extVar('PWD') + '/tests/resources/cache',
}