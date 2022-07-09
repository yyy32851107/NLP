from __future__ import print_function
import tensorflow as tf
from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_de_en_vocab, load_tw_vocab
from modules import *
import os, codecs
from tqdm import tqdm

class Graph:
    def __init__(self, is_training=True, vocab_len=None):
        if is_training:
            self.x        = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))
            self.y        = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))
            self.y_decoder_input = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))
        else:
            # inference
            self.x        = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))
            self.y        = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))
            self.y_decoder_input = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))

        # define decoder inputs
        self.decoder_inputs = tf.concat((tf.ones_like(self.y_decoder_input[:, :1])*2, self.y_decoder_input[:, :-1]), -1) # 2:<S>

        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
            ## Word Embedding
            self.enc_embed = get_token_embeddings(self.x,
                                                  vocab_size=vocab_len,
                                                  num_units=hp.hidden_units)

            ## Word Embedding
            self.dec_embed = get_token_embeddings(self.decoder_inputs,
                                                  vocab_size=vocab_len,
                                                  num_units=hp.hidden_units)

            # Get Vocab Embedding
            self.embeddings = get_token_embeddings(inputs=None,
                                                   vocab_size=vocab_len,
                                                   num_units=hp.hidden_units,
                                                   get_embedtable=True)

            self.weights = tf.transpose(self.embeddings)  # (d_model, vocab_size)

        # Encoder-MLM
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # Word src_masks
            src_masks_w = tf.math.equal(self.x, 0)  # (N, maxlen)

            ## Word Positional Encoding
            self.enc = self.enc_embed + positional_encoding(self.enc_embed, hp.maxlen)
            self.enc = tf.layers.dropout(self.enc, hp.dropout_rate, training=is_training)

            ## Word Blocks
            for i in range(hp.num_blocks_w):
                with tf.variable_scope("num_blocks_w{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    self.enc, self.att_w = multihead_attention(queries=self.enc,
                                                                keys=self.enc,
                                                                values=self.enc,
                                                                key_masks=src_masks_w,
                                                                num_heads=hp.num_heads,
                                                                dropout_rate=hp.dropout_rate,
                                                                training=is_training,
                                                                causality=False)
                    # feed forward
                    self.enc = ff(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

        # Decoder
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            # tgt_masks
            tgt_masks = tf.math.equal(self.decoder_inputs, 0)  # (N, T2)

            ## Positional Encoding
            self.dec = self.dec_embed + positional_encoding(self.dec_embed, hp.maxlen)
            self.dec = tf.layers.dropout(self.dec, hp.dropout_rate, training=is_training)

            # Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    self.dec, _ = multihead_attention(queries=self.dec,
                                                        keys=self.dec,
                                                        values=self.dec,
                                                        key_masks=tgt_masks,
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        training=is_training,
                                                        causality=True,
                                                        scope="self_attention")

                    # Vanilla attention
                    self.dec, self.att_v = multihead_attention(queries=self.dec,
                                                                keys=self.enc,
                                                                values=self.enc,
                                                                key_masks=src_masks_w,
                                                                num_heads=hp.num_heads,
                                                                dropout_rate=hp.dropout_rate,
                                                                training=is_training,
                                                                causality=False,
                                                                scope="vanilla_attention")
                    ### Feed Forward
                    self.dec = ff(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])

            # Final linear projection (embedding weights are share
            self.weights = tf.transpose(self.embeddings)                   # (d_model, vocab_size)
            self.logits = tf.einsum('ntd,dk->ntk', self.dec, self.weights)  # (N, T_q, vocab_size)
            #self.logits = tf.layers.dense(self.dec, vocab_len)            # (N, T_q, tw_vocab_size)

        if is_training:
            # Loss_context
            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=vocab_len))  # (N, T_q, vocab_size)
            self.ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
            self.nonpadding = tf.to_float(tf.not_equal(self.y, 0))  # 0: <pad> #(N,T_q)
            self.loss = tf.reduce_sum(self.ce * self.nonpadding) / (tf.reduce_sum(self.nonpadding) + 1e-7)

            # Loss
            self.global_step = tf.train.get_or_create_global_step()
            self.lr = noam_scheme(hp.lr, self.global_step, hp.warmup_steps)
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            #var_list = tf.get_collection(tf.GraphKeys.VARIABLES, scope='encoder|decoder')
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            # inference
            self.prob = tf.nn.softmax(self.logits)  # (N, T_q, vocab_size)
            self.preds = tf.to_int32(tf.argmax(self.prob, axis=-1))  # (N, T_q)

            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=vocab_len))  # (N, T_q, vocab_size)
            self.ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=self.y_smoothed)
            self.noncost_unk = tf.to_float(tf.not_equal(self.y, 1))  # 1: <unk>
            self.noncost_pad = tf.to_float(tf.not_equal(self.y, 0))  # 0: <pad>
            self.noncost = self.noncost_unk * self.noncost_pad
            self.los = (tf.reduce_sum(self.ce * self.noncost) / tf.reduce_sum(self.noncost))
            self.ppl = 2**(self.los)  # (N, T_q)


if __name__ == '__main__':
    # Load vocabulary
    token2idx, idx2token = load_de_en_vocab()
    token2idx_len = len(token2idx)
    X, Y, Y_DI, num_batch = get_batch_data()

    graph = tf.Graph()
    with graph.as_default():
        # Construct graph
        g = Graph(True, token2idx_len)
        saver = tf.train.Saver()
        print("Graph loaded")

        # Start session
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            if hp.curriculum == hp.phase_two:
                # load checkpoint
                var_to_restore = tf.contrib.framework.get_variables_to_restore(include=['encoder','decoder'])
                saver = tf.train.Saver(var_to_restore)
                saver.restore(sess, tf.train.latest_checkpoint(hp.logdir + str(hp.phase_one)))
                print("en_de_var_restored")
                saver = tf.train.Saver()
                # Update saver is must!
                # Otherwise,the next saved model will only have the parameters loaded into the above saver

            for epoch in range(1, hp.num_epochs+1):
                loss=[]
                for step in tqdm(range(num_batch), total=num_batch, ncols=100, unit='b'):
                    x =               X[step * hp.batch_size: (step + 1) * hp.batch_size]
                    y =               Y[step * hp.batch_size: (step + 1) * hp.batch_size]
                    y_decoder_input = Y_DI[step * hp.batch_size: (step + 1) * hp.batch_size]

                    _, loss_step = sess.run([g.train_op, g.loss],
                                            {g.x:x, g.y:y, g.y_decoder_input:y_decoder_input})

                    loss.append(loss_step)

                print("epoch:%03d train_loss:%.5lf\n"%(epoch, np.mean(loss)))

                #if epoch in [60,70,80,90,100]:
                if epoch in [80, 100, 120, 130, 150]:
                    gs = sess.run(g.global_step)
                    saver.save(sess, hp.logdir + str(hp.curriculum) + '/model_epoch_%02d_gs_%d' % (epoch, gs))

print("Train Done")
