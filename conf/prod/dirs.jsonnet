snli:{
    dev:std.extVar('PWD') + '/data/snli_1.0/snli_1.0_dev.jsonl',
    train:std.extVar('PWD') + '/data/snli_1.0/snli_1.0_train.jsonl',
    test:std.extVar('PWD') + '/data/snli_1.0/snli_1.0_test.jsonl',
}, 
cache_path:{
    train:std.extVar('PWD') + '/data/cache/train',
    dev:std.extVar('PWD') + '/data/cache/dev',
    test:std.extVar('PWD') + '/data/cache/test',
}