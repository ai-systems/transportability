snli:{
    dev:std.extVar('PWD') + '/data/snli_1.0/multinli_1.0_dev_matched.jsonl',
    train:std.extVar('PWD') + '/data/snli_1.0/multinli_1.0_train.jsonl',
    test:std.extVar('PWD') + '/data/snli_1.0/snli_1.0_test.jsonl',
}, 
cache_path:{
    train:std.extVar('PWD') + '/data/cache/train_mutli',
    dev:std.extVar('PWD') + '/data/cache/dev_multi',
    test:std.extVar('PWD') + '/data/cache/test_multi',
}