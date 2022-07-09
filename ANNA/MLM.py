import numpy as np

def mask_tokens(inputs, mlm_probability, vocab_len, special_tokens_mask):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = np.copy(inputs)
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = np.random.random_sample(labels.shape)
    special_tokens_mask = special_tokens_mask.astype(np.bool_)

    probability_matrix[special_tokens_mask] = 0.0
    masked_indices = probability_matrix > (1 - mlm_probability)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    j_mat = np.not_equal(inputs, 0)  # regardless pad

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (np.random.random_sample(labels.shape) < 0.8) & masked_indices & j_mat
    inputs[indices_replaced] = 1 #<UNK> -- [MASK]

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (np.random.random_sample(labels.shape) < 0.5) & masked_indices & ~indices_replaced & j_mat
    random_words = np.random.randint(low=4, high=vocab_len, size=np.count_nonzero(indices_random), dtype=np.int32)
    inputs[indices_random] = random_words

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    labels = np.where(j_mat, labels, -100)  # replaced pad to -100

    return inputs, labels