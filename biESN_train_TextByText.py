import scipy.sparse.linalg
import cPickle
import argparse
import os

import tensorflow as tf

import loss_functions

from numpy import flip, random, linalg, zeros, tanh, dot, vstack, sqrt, eye, asarray, reshape, sum

from preprocess_data import load_embeddings, read_data, format_data, get_lemma_synset_maps, get_lemma2syn


if __name__ == "__main__":

    parser = argparse.ArgumentParser(version='1.0', description='Train or evaluate a neural WSD model.',
                                     fromfile_prefix_chars='@')
    parser.add_argument('-embeddings_model', dest='embeddings_model', required=True, help='Location of the embeddings.')
    parser.add_argument('-train_data', dest='train_data', required=True, help='Path to the training data.')
    parser.add_argument('-sensekey2synset', dest='sensekey2synset', required=True, help='Path to the synset mappings.')
    parser.add_argument('-dictionary', dest='dictionary', required=True, help='Path to the WordNet dictionary.')
    parser.add_argument('-embeddings_size', dest='embeddings_size', required=True)
    parser.add_argument('-window_size', dest='window_size', required=True)
    parser.add_argument('-use_reservoirs', dest='use_reservoirs', required=False, default="True",
                        help="Use reseroirs or train directly on the word embeddings.")
    parser.add_argument('-softmax', dest='softmax', required=False, default="False",
                        help="Use a softmax classifier at the end of the network.")
    parser.add_argument('-learning_rate', dest='learning_rate', required=False, default=1,
                        help="Learning rate in case the Tensorflow softmax implementation is used.")
    parser.add_argument('-bidirectional', dest='bidirectional', required=False, default="True",
                        help="Use a bidirectional architecture, or just one reservoir.")
    parser.add_argument('-res_size', dest='res_size', required=False, default=100,
                        help="Size of the echo state reservoirs.")
    parser.add_argument('-leak_rate', dest='leak_rate', required=False, default=1, help="Leaking rate.")
    parser.add_argument('-res_sparsity', dest='res_sparsity', required=False, default=0.5, help="Reservoir sparsity.")
    parser.add_argument('-only_open_class', dest='only_open_class', required=False, default="True")
    parser.add_argument('-training_iterations', dest='training_iterations', required=False, default="1")
    parser.add_argument('-save_path', dest='save_path', required=True)

    args = parser.parse_args()
    embeddings_model = args.embeddings_model
    train_data_path = args.train_data
    sensekey2synset = args.sensekey2synset
    dictionary = args.dictionary
    embeddings_size = int(args.embeddings_size)
    window_size = int(args.window_size)
    use_reservoirs = args.use_reservoirs
    softmax = args.softmax
    learning_rate = float(args.learning_rate)
    bidirectional = args.bidirectional
    if use_reservoirs == "True":
        res_size = int(args.res_size)
    else:
        res_size = 0
    if bidirectional == "True":
        total_res_size = res_size * 2
    else:
        total_res_size = res_size
    a = float(args.leak_rate)  # leaking rate
    res_sparsity = float(args.res_sparsity)
    only_open_class = args.only_open_class
    training_iterations = int(args.training_iterations)
    save_path = args.save_path

    embeddings = load_embeddings(embeddings_model)  # load the embeddings
    f_sensekey2synset = cPickle.load(open(sensekey2synset, "rb"))  # get the mapping between synset keys and IDs
    train_data, known_lemmas = read_data(train_data_path, f_sensekey2synset, only_open_class)  # read the training data
    lemma2synset = get_lemma2syn(dictionary)
    lemma2id, synset2id = get_lemma_synset_maps(lemma2synset, known_lemmas)

    if softmax == "True":
        output_size = len(synset2id)
    else:
        output_size = embeddings_size
    input_size = embeddings_size * window_size

    # generate the ESN reservoir
    name_add = str(res_size) + '_' + str(res_sparsity) + '_' + str(a)
    random.seed(42)
    Wout = (random.rand(output_size, total_res_size + input_size) - 0.5) * 1.0
    if softmax == "True":
        biases = asarray(reshape((random.rand(output_size) - 0.5) * 1.0, [1, output_size]))
    if use_reservoirs == "True":
        Win_fw = (random.rand(res_size, input_size) - 0.5) * 1
        Win_bw = (random.rand(res_size, input_size) - 0.5) * 1
        Wini = scipy.sparse.rand(res_size, res_size, density=res_sparsity)
        i, j, v = scipy.sparse.find(Wini)
        # W = Wini.toarray()
        W_fw = Wini.toarray()
        W_bw = Wini.toarray()
        # W[i, j] -= 0.5
        W_fw[i, j] -= 0.5
        W_bw[i, j] -= 0.5

        # normalizing and setting spectral radius
        print 'Computing spectral radius...'
        # rhoW = max(abs(linalg.eig(W)[0]))
        rhoW_fw = max(abs(linalg.eig(W_fw)[0]))
        rhoW_bw = max(abs(linalg.eig(W_bw)[0]))
        print 'done.'
        # W *= 1.25 / rhoW
        W_fw *= 1.25 / rhoW_fw
        W_bw *= 1.25 / rhoW_bw

    all_train_error = []

    # RLS training
    print 'Calculating reservoir states...'
    if softmax == "True":
        session = tf.Session()
        softmax_model = loss_functions.SoftmaxModel(total_res_size + input_size, output_size, learning_rate)
        init = tf.global_variables_initializer()
        feed_dict = { softmax_model.weights_placeholder: Wout.T,
                      softmax_model.biases_placeholder: biases}
        session.run(init, feed_dict=feed_dict)
    else:
        RLS_delta = 0.000001
        RLS_lambda = 0.9999995
        SInverse = (1.0 / RLS_delta) * eye(total_res_size + input_size)
    intermediate_data = []
    for i,_ in enumerate(train_data):
        print 'calculating reservoir states on text ' + str(i+1) + '/' + str(len(train_data)) + '...'
        inputs, fw_states, bw_states = [], [], []
        Xtr, Ytr,_, _, _, _ = format_data(train_data[i], embeddings, embeddings_size, output_size, window_size,
                                          synset2id, softmax)
        trainLen = len(Xtr)
        u_fw = zeros((input_size, 1))
        if use_reservoirs == "True":
            u_bw = zeros((input_size, 1))
            x_fw = zeros((res_size, 1))
            x_bw = zeros((res_size, 1))
            state_fw = zeros((res_size + input_size, 1))
            state_bw = zeros((res_size + input_size, 1))
        for t in range(trainLen):
            u_fw = reshape(asarray(Xtr[t]), (input_size, 1))
            inputs.append(u_fw)
            if use_reservoirs == "True":
                u_bw = reshape(asarray(Xtr[trainLen-1-t]), (input_size, 1))
                x_fw = (1 - a) * x_fw + a * tanh(dot(Win_fw, u_fw) + dot(W_fw, x_fw))
                x_bw = (1 - a) * x_bw + a * tanh(dot(Win_bw, u_bw) + dot(W_bw, x_bw))
                fw_states.append(x_fw)
                bw_states.append(x_bw)
        text = (inputs, fw_states, flip(bw_states, 0), Ytr)

        print 'training on text ' + str(i + 1) + '/' + str(len(train_data)) + '...'
        trainLen = len(text[0])
        curr_train_error = zeros((trainLen))
        Ytr = text[3]
        e = zeros((output_size, 1))
        for t in range(trainLen):
            u = text[0][t]
            if use_reservoirs == "True":
                x_fw = text[1][t]
                if bidirectional == "True":
                    x_bw = text[2][t]
                    state = vstack((u, x_fw, x_bw))
                else:
                    state = vstack((u, x_fw))
            else:
                state = u
            gold_label = reshape(Ytr[t], (output_size, 1))
            if softmax == "True":
                feed_dict = {softmax_model.reservoir_states: state.T,
                             softmax_model.labels: gold_label.T
                             }
                ops = [softmax_model.train_op]
                fetches = session.run(ops, feed_dict=feed_dict)
            else:
                y = dot(Wout, state)
                phi = dot(state.T, SInverse)
                k = phi.T / (RLS_lambda + dot(phi, state))
                e = gold_label - y
                curr_train_error[t] = sqrt(sum(dot(e, e.T))) / (len(e))
                Wout += k.T * e
                SInverse = (SInverse - dot(k, phi)) / RLS_lambda
        all_train_error.append(curr_train_error)

    print "...done."

    print 'Saving...'
    with open(os.path.join(save_path, 'trained_ESN3_' + name_add + '_parameters.txt'), 'w') as params:
        params.write(str(args) + '\n\n')
    f = open(os.path.join(save_path, 'trained_ESN3_' + name_add + '.cpickle'), "wb")
    cPickle.dump(all_train_error, f, protocol=2)
    cPickle.dump(res_sparsity, f, protocol=2)
    cPickle.dump(a, f, protocol=2)
    if softmax == "True":
        saver = tf.train.Saver()
        saver.save(session, os.path.join(args.save_path, "model/model.ckpt"))
        cPickle.dump(synset2id, f, protocol=2)
        cPickle.dump(learning_rate, f, protocol=2)
    else:
        cPickle.dump(Wout, f, protocol=2)
    if use_reservoirs == "True":
        cPickle.dump(Win_fw, f, protocol=2)
        cPickle.dump(Win_bw, f, protocol=2)
        cPickle.dump(W_fw, f, protocol=2)
        cPickle.dump(W_bw, f, protocol=2)
    f.close()