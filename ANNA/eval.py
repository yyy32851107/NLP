from __future__ import print_function
import codecs
import os
import argparse
import tensorflow as tf
import numpy as np
from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_en_vocab, load_tw_vocab
from train import Graph

def eval():
    # Load vocabulary
    token2idx, idx2token = load_de_en_vocab()
    token2idx_len = len(token2idx)

    # Load data
    X, Y, Sources, Targets = load_test_data()

    graph = tf.Graph()
    with graph.as_default():
        # Construct graph
        g = Graph(False, token2idx_len)
        saver = tf.train.Saver()

        """
        reader = tf.train.NewCheckpointReader('./Check_point2/model_epoch_100_gs_156400')
        var_to_shape_map = reader.get_variable_to_shape_map()
        for var_name in var_to_shape_map.keys():
            if var_name.startswith('music_encoder'):
                var_value = reader.get_tensor(var_name)
                print("var_name", var_name)
                print("var_value", var_value)
        """
        ## Get model name
        mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

        # Start session
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Graph Restored")

            ## Inference
            if not os.path.exists('results'): os.mkdir('results')

            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                print("Evaluation Start")
                #args = parse_args()
                #if args.a:
                #    att_f = codecs.open("results/" + "attention_vis", "w", "utf-8")
                #ppls = []
                for i in range(len(X) // hp.batch_size):
                    ### Get mini-batches
                    x =              X[i * hp.batch_size: (i + 1) * hp.batch_size]
                    y =              Y[i * hp.batch_size: (i + 1) * hp.batch_size]
                    sources =  Sources[i * hp.batch_size: (i + 1) * hp.batch_size]
                    targets =  Targets[i * hp.batch_size: (i + 1) * hp.batch_size]

                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds,{g.x:x, g.y:y, g.y_decoder_input:preds})
                        preds[:, j] = _preds[:, j]

                    ### Write to file
                    #for source, target, pred, att_w, att_u in zip(sources, targets, preds, att_ws, att_us): # sentence-wise
                    for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                        got = " ".join(idx2token[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- source: " + source +"\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()
                        #if args.a:
                        #    att_f.write("- att_w: " + str(att_w) + "\n")
                        #    att_f.write("- att_u: " + str(att_u) + "\n\n\n\n\n")
                        #    att_f.flush()

            print("Results are written ")
            #if args.p:
                #print("- ppl_score: " + "".join('%s' % np.mean(ppls)))
            #print(ppls)

#def parse_args():
    #parser = argparse.ArgumentParser("evaluate_option")
    #parser.add_argument("--a", action="store_true")
    #parser.add_argument("--p", action="store_true")
    #return parser.parse_args()

if __name__ == '__main__':
    eval()
    print("Evaluation completed")
