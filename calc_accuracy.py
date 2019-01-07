import os

from sklearn.metrics.pairwise import cosine_similarity

pos2pos = {"ADJ":"a", "ADV":"r", "NOUN":"n", "VERB":"v"}

def get_lemma2syn(f_dictionary):
    dictionary = open(f_dictionary, "r").readlines()
    lemma2syn = {}
    for line in dictionary:
        fields = line.strip().split(" ")
        lemma, synsets = fields[0], [syn[:10] for syn in fields[1:]]
        lemma2syn[lemma] = synsets
    return lemma2syn

def calculate_accuracy(outs, gold_synsets, lemmas, pos_filters, term_ids, embeddings, dictionary, syn2gloss):
    lemma2syn = get_lemma2syn(dictionary)
    count_correct = 0
    count_all = 0
    unavailable_syn_emb = set()
    unavailable_syn_cases = 0
    errors = ""
    for count, gold in enumerate(gold_synsets):
        # if using the test data, use "test_lemmas", otherwise use "train_lemmas"
        lemma = lemmas[count]
        pos = pos_filters[count]
        term_id = term_ids[count]
        if lemma in lemma2syn:
            possible_syns = lemma2syn[lemma]
        else:
            count_all += 1
            continue
        output = outs[count]
        max_sim = -10000.0
        selected_syn = ""
        for syn in possible_syns:
            if pos2pos[pos] != syn.split("-")[1]:
                continue
            if syn in embeddings:
                cos_sim = cosine_similarity(output.reshape(1,-1), embeddings[syn].reshape(1,-1))[0][0]
            else:
                unavailable_syn_cases += 1
                unavailable_syn_emb.add(syn)
                cos_sim = 0.0
            if cos_sim > max_sim:
                max_sim = cos_sim
                selected_syn = syn
        # gold_cos_sim = cosine_similarity(output.reshape(1,-1), embeddings[gold].reshape(1,-1))[0][0]
        # line_to_write = lemma + "\t" + selected_syn + "\t" + gold + "\t" + str(max_sim - gold_cos_sim) + "\n"
        # out.write(line_to_write)
        if selected_syn in gold:
            count_correct += 1
        else:
            if syn2gloss is not None:
                gold_ids, gold_glosses = [], []
                for s in gold:
                    gold_ids.append(s)
                    gold_glosses.append(syn2gloss[s])
                gold_choice = "||".join(gold_ids)
                gold_glosses = "||".join(gold_glosses)
                error_log = term_id + "\t" + lemma + "\t" + gold_choice + "\t" + gold_glosses + "\t" + selected_syn + "\t" + \
                            syn2gloss[selected_syn] + "\t" + str(max_sim) + "\n"
                errors += error_log
        count_all += 1
    return count_correct, count_all, errors