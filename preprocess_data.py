import _elementtree
import gensim
import numpy
import os
import pickle
import copy



def load_embeddings(embeddings_path, binary=False):
    _, extension = os.path.splitext(embeddings_path)
    if extension == ".txt":
        binary = False
    elif extension == ".bin":
        binary = True
    embeddings_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=binary,
                                                                       datatype=numpy.float32)
    return embeddings_model

# Read the data from SemCor & WSDEVAL, write it into a list of sentences
# Each sentence contains a list of word and each word is a list of features: [wordform, lemma, pos, [synset(s)]]
def read_data(path, sensekey2synset, only_open_class="True"):
    data = []
    path_data = ""
    path_keys = ""
    for f in os.listdir(path):
        if f.endswith(".xml"):
            path_data = f
        elif f.endswith(".txt"):
            path_keys = f
    codes2keys = {}
    f_codes2keys = open(os.path.join(path, path_keys), "r")
    for line in f_codes2keys.readlines():
        fields = line.strip().split()
        code = fields[0]
        keys = fields[1:]
        codes2keys[code] = keys
    tree = _elementtree.parse(os.path.join(path, path_data))
    corpus = tree.getroot()
    for text in corpus:
        text_data = []
        sentences = text.findall("sentence")
        for sentence in sentences:
            current_sentence = []
            elements = sentence.findall(".//")
            for element in elements:
                id = element.get("id")
                pos = element.get("pos")
                if only_open_class == "True" and pos not in ["NOUN", "VERB", "ADJ", "ADV"]:
                    continue
                wordform = element.text
                lemma = element.get("lemma")
                if element.tag == "instance":
                    synsets = [sensekey2synset[key] for key in codes2keys[element.get("id")]]
                else:
                    synsets = None
                current_sentence.append([wordform, lemma, pos, synsets, id])
            text_data.append(current_sentence)
        data.append(text_data)
    return data

def construct_contexts(sentences, window_size):
    ctx_shift = window_size / 2
    contexts = []
    gold_data = []
    lemmas = []
    synsets = []
    pos = []
    term_id = []
    for sentence in sentences:
        sent_ctx = []
        sent_gold = []
        for i, word in enumerate(sentence):
            if word[-1] is not None:
                ctx = []
                for n in range(ctx_shift+1):
                    if n == 0:
                        ctx.append(word[1])
                    else:
                        if i+n < len(sentence):
                            ctx.append(sentence[i+n][1])
                        else:
                            ctx.append("NULL")
                        if i-n >= 0:
                            ctx.insert(0, sentence[i-n][1])
                        else:
                            ctx.insert(0, "NULL")
                sent_ctx.append(ctx)
                sent_gold.append(word[-1])
                lemmas.append(word[1])
                synsets.append(word[3])
                pos.append(word[2])
                term_id.append(word[4])
        contexts.append(sent_ctx)
        gold_data.append(sent_gold)
    return contexts, gold_data, lemmas, synsets, pos, term_id

def format_data(sentences, embeddings, EMBEDDINGS_SIZE, CONTEXT_WINDOW_SIZE):
    input_vectors = []
    gold_vectors = []
    lemmas = []
    synsets = []
    pos = []
    term_id = []
    for sentence in sentences:
        contexts, gold_labels, lemmas_sent, synsets_sent, pos_sent, term_id_sent = construct_contexts([sentence], CONTEXT_WINDOW_SIZE)
        lemmas.extend(lemmas_sent)
        synsets.extend(synsets_sent)
        pos.extend(pos_sent)
        term_id.extend(term_id_sent)
        zero = numpy.zeros(EMBEDDINGS_SIZE, dtype=numpy.float32)
        for i, sent_context in enumerate(contexts):
            for j, ctx in enumerate(sent_context):
                ctx_vectors = []
                for word in ctx:
                    if word in embeddings:
                        ctx_vectors.append(embeddings[word])
                    else:
                        ctx_vectors.append(zero)
                input_vector = numpy.concatenate(ctx_vectors, 0)
                curr_gold_labes = gold_labels[i][j]
                gold_vector = copy.copy(zero)
                for synset in curr_gold_labes:
                    if synset in embeddings:
                        gold_vector += embeddings[synset]
                    # else:
                    #     print synset
                gold_vector /= len(curr_gold_labes)
                gold_vectors.append(gold_vector)
                input_vectors.append(input_vector)
    return input_vectors, gold_vectors, lemmas, synsets, pos, term_id





