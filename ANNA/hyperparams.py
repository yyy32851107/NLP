class Hyperparams:
    '''Hyperparameters'''
    # data
    source_target_vocab = '../cop/train_data-en/corpra_dev_test_final_turn_len50_turn15.txt'
    topic_vocab = '../cop/train_data-en/corpra_dev_test_final_tp.txt'


    source_train = '../cop/train_data-en/corpra_dev_test_final_s.txt'
    target_train = '../cop/train_data-en/corpra_dev_test_final_t.txt'
    topic_train = '../cop/train_data-en/corpra_dev_test_final_tp.txt'


    source_test = '../cop/test_data/corpra_dev_test_final_s.txt'
    target_test = '../cop/test_data/corpra_dev_test_final_t.txt'
    topic_test = '../cop/test_data/corpra_dev_test_final_tp.txt'


    # training
    batch_size = 128 # alias = N
    lr = 0.0002 # learning rate. In paper, learning rate is adjusted to the global step.
    warmup_steps = 4000
    logdir = 'Check_point' # log directory
    phase_one = 1
    phase_two = 2

    # model
    maxlen = 51 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    gener_maxlen = 40   # Maximum number of generated response words.
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks_w = 4 # number of encoder/decoder blocks on word level
    num_blocks = 4
    num_epochs = 100
    num_heads = 8
    dropout_rate = 0.1

    curriculum = 1