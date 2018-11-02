K = 1  # Unk frequency limit
import collections
import numpy as np
import sys

class Sentence(object):
    def __init__(self, tokens=None, tags=None, pred_tags=None):
        self.tokens = tokens
        self.tags = tags
        self.pred_tags = pred_tags


def load_and_preprocess(fname):
    txt_data_tmp = open(fname).read().strip().split("\n")
    txt_data = []
    word_frq_count = collections.Counter()
    for sentence in txt_data_tmp:
        tokens, tags = [], []
        tokens_and_tags = sentence.split(' ')
        for pair in tokens_and_tags:
            token_and_tag = pair.split("/")
            tokens.append(token_and_tag[0].lower())
            tags.append(token_and_tag[1])
        word_frq_count.update(tokens)  # word frq update
        txt_data.append(Sentence(tokens, tags))
    print("Train data size".format(len(txt_data)))
    # repalce less frequent words with 'Unk'
    less_frq_set = set()
    vocab = set()
    for word, count in word_frq_count.items():
        if count <= K:
            less_frq_set.add(word)
        else:
            vocab.add(word)
    for sentence in txt_data:
        for idx, token in enumerate(sentence.tokens):
            if token in less_frq_set:
                sentence.tokens[idx] = 'Unk'

    #token is repalced with 'Unk'
    vocab.add('Unk')
    print("Vocabulary size: {}, with {} less frequent words".format(len(vocab), len(less_frq_set)))

    return txt_data,vocab, word_frq_count, less_frq_set  # retured arrays are the word and corresponding tags


"""
def count_num_and_unk(txt_data,pos_tag):
    word_frq_count = collections.defaultdict(int)
    inst_num = len(txt_data)

    for i in range(inst_num):
        word = txt_data[i]
        word_frq_count[word] += 1

    idx = 0
    # will map works with low frequencies as Unk and the tag is: UNK
    for i in range(inst_num):
        word = txt_data[i]
        count = word_frq_count[word]
        if count <= K:
            pos_tag[i] = 'Unk'
    print("vocab size:", len(set(txt_data)))
    return txt_data, pos_tag, word_frq_count
"""

def construct_idx(vocab):
    tag2idx = {'START': 0, 'A': 1, 'C': 2, 'D': 3, 'M': 4, 'N': 5,
               'O': 6, 'P': 7, 'R': 8, 'V': 9, 'W': 10, 'END': 11}  # to construct the transition matrix
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}  # to save tag into CSV file
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return tag2idx, idx2tag, word2idx, idx2word


def trans_count_cal(trn_data, tag2idx):
    tag_num = len(tag2idx)
    trans_count = np.zeros((tag_num, tag_num))
    for data in trn_data:
        seq_len = len(data.tokens) + 1
        for i in range(seq_len):
            if i == 0:
                # first tag, start with START
                prev_tag, curr_tag = 'START', data.tags[i]
            elif i == seq_len-1:
                # last tag to END
                prev_tag, curr_tag = data.tags[i-1], 'END'
            else:
                prev_tag, curr_tag = data.tags[i-1], data.tags[i]
            trans_count[tag2idx[prev_tag], tag2idx[curr_tag]] += 1

    return trans_count


def ems_count_cal(trn_data, word2idx, tag2idx):
    word_num = len(word2idx)
    real_tag_num = len(tag2idx) - 2
    # ems count matrix, where start and end does not exist here
    ems_count = np.zeros((real_tag_num, word_num))
    for data in trn_data:
        seq_len = len(data.tokens)
        for i in range(seq_len):
            word, tag = data.tokens[i], data.tags[i]
            ems_count[tag2idx[tag]-1, word2idx[word]] += 1
    return ems_count


def viterbi_decode(data, trans_prob, ems_prob, word2idx, tag2idx, idx2tag):
    # note for processing convenience, ems is V*(K-2), trans prob is K*K

    #preprocessing can avoid

    tag_num = len(tag2idx)
    for sentence in data:
        word_num = len(sentence.tokens)
        v_m = np.zeros((word_num + 1, tag_num))
        b_m = np.zeros((word_num + 1, tag_num),dtype = int)
        #v = np.zeros(word_num + 1)
        for i in range(word_num+1):
            if i < word_num:
                token = sentence.tokens[i] if sentence.tokens[i] in word2idx else 'Unk'
            if i == 0:#v_1 case:
                for j in range(len(tag2idx)):
                    if j == 0:  # there is no workd should end at 'START'
                        tranp = -1e20
                        emsp = -1e20
                    elif j == len(tag2idx) - 1:
                        tranp = -1e20
                        emsp = -1e20
                    else:
                        tranp = trans_prob[0,j]
                        emsp = ems_prob[j-1,word2idx[token]]
                    v_m[i, j] = emsp + tranp
                b_m[i, :] = 0
            elif i == word_num:
                # tag END is observed
                for j in range(len(tag2idx)):
                    if j == 0:
                        tranp = -1e20
                        emsp = 0
                    elif j == len(tag2idx) - 1:  # there is no transition from END to END
                        tranp = -1e20
                        emsp = 0
                    else:
                        tranp = trans_prob[j, len(tag2idx) - 1]
                        emsp = 0
                    v_prev = v_m[i-1, j]
                    v_m[i, j] = tranp + emsp + v_prev
                b_m[i, :] = np.argmax(v_m[i, :])  # for the END tag, it takes same prev_tag for END
            else:
                # normal treatment of viterbi decoding
                for j in range(len(tag2idx)):
                    if j == 0:  # START and END should not appear in the middle
                        tranp = -1e20
                        emsp = -1e20
                        v_prev = -1e20
                        v_m[i, j] = tranp + emsp + v_prev
                    elif j == len(tag2idx) - 1:
                        tranp = -1e20
                        emsp = -1e20
                        v_prev = -1e20
                        v_m[i, j] = tranp + emsp + v_prev
                    else:
                        # K^2 compuatations needed
                        #total_score = []
                        ep = ems_prob[j-1, word2idx[token]]
                        #for k in range(len(tag2idx)):
                        tp = trans_prob[:,j]
                        v_prev_tmp = v_m[i-1,:]
                        #total_score.append(tp + ep + v_prev_tmp)
                        total_score = tp + ep + v_prev_tmp
                        v_m[i, j] = np.amax(total_score)
                        b_m[i, j] = np.argmax(total_score)
        decode_seq = []
        for i in range(word_num + 1):
            # decode_seq.append(np.argmax(v_m[word_num-i,:]))
            if i == 0:
                prev_idx = b_m[word_num - i, 0]
                tag_idx = idx2tag[prev_idx]
            else:
                prev_idx = b_m[word_num-i, prev_idx]
                tag_idx = idx2tag[prev_idx]
            if i < word_num:
                decode_seq.insert(0,tag_idx)
        sentence.pred_tags = decode_seq


def cal_emit_count_jc(trn_data, word2idx,tags2idx):
    ''' caculate emission count with shape = (V, K)
    '''
    V = len(word2idx)
    K = len(tags2idx)
    emit_count = np.zeros((V, K))
    for sent in trn_data:
        sent_len = len(sent.tokens)
        for idx in range(sent_len):
            token, tag = sent.tokens[idx], sent.tags[idx]
            widx, tidx = word2idx[token], tags2idx[tag]
            emit_count[widx][tidx] += 1
    return emit_count


def viterbi_decode_jc(data, smooth_trans_logprobs, smooth_emit_logprobs, word2idx,tags2idx,idx2tags):
    ''' Viterbi_decoding algorithm
    '''
    K1 = len(tags2idx)
    for sent in data:
        # intialization
        pred_tags = []
        sent_len = len(sent.tokens)
        V = np.zeros((sent_len, K1)) # v shape = (sent_len, k)
        B = np.zeros((sent_len, K1), dtype=int) # b shape = (sent_len, k) or (sent_len-1, k)
        # v1
        token = sent.tokens[0] if sent.tokens[0] in word2idx else 'Unk'
        V[0,:] = smooth_trans_logprobs[tags2idx['START'], :] + smooth_emit_logprobs[word2idx[token],:]
        B[0,:] = np.argmax(smooth_trans_logprobs[tags2idx['START'], :] + smooth_emit_logprobs[word2idx[token],:])
        # forward
        for idx in range(1, sent_len):
            token = sent.tokens[idx] if sent.tokens[idx] in word2idx else 'Unk'
            # V_prev
            for k in range(K):
                scores = smooth_trans_logprobs[:,k] + smooth_emit_logprobs[word2idx[token],:] + V[idx-1, :]
                V[idx, k] = max(scores)
                B[idx, k] = np.argmax(scores)
        scores = smooth_trans_logprobs[:,tags2idx['END']] + V[-1, :]
        y = np.argmax(scores)
        # backward
        for idx in range(sent_len-1, -1, -1):
            pred_tags.insert(0, idx2tags[B[idx, y]])
            y = B[idx, y]
        sent.pred_tags = pred_tags
        # assert len(sent.preds) == len(sent.tags)

def save_trans_prob(trans_prob, idx2tag,fname):
    with open(fname, 'w') as file:
        for prev_idx in range(trans_prob.shape[0]-1):
            for curr_idx in range(trans_prob.shape[1]-1):
                prev_tag, curr_tag = idx2tag[prev_idx], idx2tag[curr_idx+1]
                file.write(','.join([prev_tag, curr_tag, str(trans_prob[prev_idx, curr_idx+1])]) + '\n')


def save_ems_prob(ems_prob, idx2word, idx2tag,fname):
    with open(fname, 'w') as file:
        for curr_idx in range(ems_prob.shape[1]):
            for prev_idx in range(ems_prob.shape[0]):
                #it is more reable to assign tag probs for each word
                tag, word = idx2tag[prev_idx+1], idx2word[curr_idx]
                file.write(','.join([tag, word, str(ems_prob[prev_idx, curr_idx])]) + '\n')


def load_dev_data(fname, vocab, less_frq_set):
    # both unrecognized and low frequenrcy words are converted to Unk
    dev_data_tmp = open(fname).read().strip().split("\n")
    dev_data = []

    for sentence in dev_data_tmp:
        tokens, tags = [], []
        tokens_and_tags = sentence.split(' ')
        for pair in tokens_and_tags:
            token_and_tag = pair.split("/")
            tokens.append(token_and_tag[0].lower())
            #if token_and_tag[0] in vocab and token_and_tag[0] not in less_frq_set:
            tags.append(token_and_tag[1])
            #else:
            #tags.append('Unk')
        dev_data.append(Sentence(tokens, tags))
    print("Dev data size {}".format(len(dev_data)))
    return dev_data


def load_tst_data(fname):
    tst_data_tmp = open(fname).read().strip().split("\n")
    tst_data = []

    for sentence in tst_data_tmp:
        tokens = []
        words = sentence.split(' ')
        for word in words:
            token = word.split(" ")
            tokens.append(token[0].lower())
        tst_data.append(Sentence(tokens))
    print("test data size {}".format(len(tst_data)))
    return tst_data


def pred_acc(data):
    pred_tags, tags = [], []
    for sentence in data:
        tags.extend(sentence.tags)
        pred_tags.extend(sentence.pred_tags)
    count = 0
    for i in range(len(tags)):
        if pred_tags[i] == tags[i]:
            count += 1
    return count / len(tags)

def save_test_pos(tst_data, fname):
    with open(fname, 'w') as file:
        for data in tst_data:
            tokens, preds = data.tokens, data.pred_tags
            token_and_tags = ['%s/%s' % (token, pred) for token, pred in zip(tokens, preds)]
            file.write(' '.join(token_and_tags) + '\n')
    print('test set pos is written!')

def main():
    BETA = 1
    ALPHA = 1
    trn_data, vocab, word_frq_count, less_frq_set = load_and_preprocess(fname = 'trn.pos')
    tag2idx, idx2tag, word2idx, idx2word = construct_idx(vocab)
    ##calculate transitional probability
    print("Calculating and saving transitional probability")
    trans_count = trans_count_cal(trn_data,tag2idx)# Careful: last row is zero, first column is zero
    trans_count_total = np.sum(trans_count, axis=1, keepdims=True)
    trans_count_total[-1,:] = 1.0 #No tag can follow 'END' tag
    trans_prob = trans_count / trans_count_total
    save_trans_prob(trans_prob, idx2tag,fname='saved_data/fs5xz-tprob.txt')

    print("Calculating and saving Smoothed transitional probability")
    sm_trans_count = trans_count + BETA
    #remove the non-existing 'END' as start tag and 'START' as following tag
    sm_trans_count[-1,:] = sm_trans_count[:,0] = 0

    sm_trans_count_total = np.sum(sm_trans_count, axis=1, keepdims=True)
    sm_trans_count_total[-1,:] = 1.0 #No tag can follow 'END' tag
    sm_trans_prob = sm_trans_count / sm_trans_count_total
    save_trans_prob(sm_trans_prob,idx2tag,fname='saved_data/fs5xz-tprob-smoothed.txt')


    ##calculate emission probability
    print("Calculating and saving emissioon probability")
    emit_cnt = ems_count_cal(trn_data, word2idx,tag2idx)#shape is (V,K-2) END START are excluded
    emit_tot_cnt = np.sum(emit_cnt, axis=1, keepdims=True)
    ems_prob = emit_cnt / emit_tot_cnt
    save_ems_prob(ems_prob, idx2word, idx2tag, fname='saved_data/fs5xz-eprob.txt')

    # smoothing
    sm_emit_count = emit_cnt + ALPHA
    sm_emit_count_total = np.sum(sm_emit_count, axis=1, keepdims=True)
    sm_emit_prob = sm_emit_count / sm_emit_count_total
    save_ems_prob(sm_emit_prob, idx2word, idx2tag, fname='saved_data/fs5xz-eprob-smoothed.txt')


    """
    #calculate emission probability with jc method
    print("Calculating and saving emission probability")
    emit_cnt = cal_emit_count_jc(trn_data, word2idx, tag2idx)  # shape is (V,K-2) END START are excluded
    emit_tot_cnt = np.sum(emit_cnt, axis=0, keepdims=True)
    emit_tot_cnt[:, 0] = emit_tot_cnt[:, -1] = 1.
    ems_prob = emit_cnt / emit_tot_cnt

    #save_ems_prob(ems_prob, idx2word, idx2tag, fname='saved_data/fs5xz-eprob.txt')

    # smoothing
    sm_emit_count = emit_cnt + ALPHA
    sm_emit_count[:, 0] = sm_emit_count[:, -1] = 0
    sm_emit_count_total = np.sum(sm_emit_count, axis=0, keepdims=True)
    sm_emit_count_total[:, 0] = sm_emit_count_total[:, -1] = 1.
    sm_emit_prob = sm_emit_count / sm_emit_count_total
    #save_ems_prob(sm_emit_prob, idx2word, idx2tag, fname='saved_data/fs5xz-eprob-smoothed.txt')
    """
    ##Viterbi Decoding
    # convert probabilities to log space
    dev_data = load_dev_data('dev.pos', vocab,less_frq_set)
    sm_trans_prob[-1,:] = sm_trans_prob[:,0] = 1e-50 # avoid zero in log
    #sm_emit_prob[:,-1] = sm_emit_prob[:,0] = 1e-50 # avoid zero in log
    sm_trans_logprob, sm_emis_logprob = np.log(sm_trans_prob), np.log(sm_emit_prob)
    # Viterbi algorithm in log space
    viterbi_decode(dev_data, sm_trans_logprob, sm_emis_logprob, word2idx, tag2idx, idx2tag)
    dev_acc = pred_acc(dev_data)
    print('Dev set accuracy:', dev_acc)
    # test on test set
    tst_data = load_tst_data('tst.word')
    viterbi_decode(tst_data, sm_trans_logprob, sm_emis_logprob, word2idx,tag2idx,idx2tag)
    save_test_pos(tst_data, fname='saved_data/fs5xz-viterbi.txt')

    print("All process complete!")
if __name__ == '__main__':
    main()
