import numpy as np
import random as rnd
import operations


# fe = open('fkeasypart4.lower','r')
# fd = open('fkdifficpart4.lower','r')
# fouts2t=open('fksrc2trg.lower','w')
# foutt2s=open('fktrg2src.lower','w')
# rnd.seed(7)


def repeatnoise(fe):
    fouts2t = []
    for e in fe:
        # create fksrc2trg
        easy = e.strip().split()
        winsize = 1 if rnd.randint(0, 1) == 0 else 2
        ndups = len(easy) // 8 if winsize == 1 else len(easy) // 11
        if ndups != 0 and len(easy) != 0:
            idces = set(np.random.choice(len(easy), size=(ndups,), replace=False))
            outsent = []
            for idx, word in enumerate(easy):
                wrd = ' '.join(easy[idx:idx + winsize])
                reword = wrd + ' ' + wrd.split()[0] if idx in idces else word
                outsent.append(reword)
            fouts2t.append(' '.join(outsent))
        elif ndups == 0:
            fouts2t.append(e.strip())

    return fouts2t


def dropnoise(fd):
    foutt2s = []
    for d in fd:
        # create fktrg2src
        diffi = d.strip().split()
        winsize = 1
        ndups = len(diffi) // 8 if winsize == 1 else len(diffi) // 11
        if ndups != 0 and len(diffi) != 0:
            idces = set(np.random.choice(len(diffi), size=(ndups,), replace=False))
            outsent = []
            for idx, word in enumerate(diffi):
                wrd = ' '.join(diffi[idx:idx + winsize])
                if idx not in idces:
                    outsent.append(wrd)
            foutt2s.append(' '.join(outsent))
        elif ndups == 0:
            foutt2s.append(d.strip())
    return foutt2s


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def wordordernoise(sents, noiseratio):
    sents = [sent.strip().split() for sent in sents]
    lengths = [len(sent) for sent in sents]
    for idx, length in enumerate(lengths):
        if length > 5:
            for it in range(int(noiseratio * length)):
                j = rnd.randint(0, length - 2)
                sents[idx][j], sents[idx][j + 1] = sents[idx][j + 1], sents[idx][j]
    return [' '.join(sent) for sent in sents]


def additive_noise(sent_batch,
                   lengths,
                   corpus,
                   ae_add_noise_perc_per_sent_low,
                   ae_add_noise_perc_per_sent_high,
                   ae_add_noise_num_sent,
                   ae_add_noise_2_grams):
    assert ae_add_noise_perc_per_sent_low <= ae_add_noise_perc_per_sent_high
    batch_size = len(lengths)
    if ae_add_noise_2_grams:
        shuffler_func = operations.shuffle_2_grams
    else:
        shuffler_func = operations.shuffle
    split_sent_batch = [
        sent.split()
        for sent in sent_batch
    ]
    length_arr = np.array(lengths)
    min_add_lengths = np.floor(length_arr * ae_add_noise_perc_per_sent_low)
    max_add_lengths = np.ceil(length_arr * ae_add_noise_perc_per_sent_high)
    for s_i in range(ae_add_noise_num_sent):
        add_lengths = np.round(
            np.random.uniform(min_add_lengths, max_add_lengths)
        ).astype(int)
        next_batch = operations.shuffle(corpus.next_batch(batch_size))
        for r_i, new_sent in enumerate(next_batch):
            addition = shuffler_func(new_sent.split())[:add_lengths[r_i]]
            split_sent_batch[r_i] += addition
    noised_sent_batch = [
        " ".join(shuffler_func(sent))
        for sent in split_sent_batch
    ]
    return noised_sent_batch


def numberfiltering(sents):
    # replace any word with numbers in it as <OOV>
    sents = [sent.strip().split() for sent in sents]
    for idx in range(len(sents)):
        for pos in range(len(sents[idx])):
            if hasNumbers(sents[idx][pos]):
                sents[idx][pos] = 'BlahBlah'
    # print(sents)
    return [' '.join(sent) for sent in sents]


