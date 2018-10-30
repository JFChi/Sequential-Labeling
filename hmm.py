from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score

K = 13
tags2idx = {'START':0, 'A':1, 'C':2, 'D':3, 'M':4, 'N':5,\
                'O':6, 'P':7, 'R':8, 'V':9, 'W':10, 'Unk': 11, 'END':12}
idx2tags = {idx: tag for tag, idx in tags2idx.items()}
print(idx2tags)
ALPHA = 1
BETA = 1

class Sent(object):
    def __init__(self, tokens=None, tags=None):
        self.tokens = tokens
        self.tags = tags
        self.pred = None

def load_train_data(fname):
    ''' load train data 
    '''
    with open(fname) as fin:
        raw_data = fin.read().splitlines()
    data = []
    vocal2count = Counter()
    for sent in raw_data:
        tokens, tags = [], []
        token_tag_pairs = sent.split(' ')
        for pair in token_tag_pairs:
            token, tag = pair.split('/')
            # token to lowercase
            token = token.lower()
            tokens.append(token), tags.append(tag)
        vocal2count.update(tokens)
        sent = Sent(tokens, tags)
        data.append(sent)
    print("Loaded {} sentences in train set".format(len(data)))
    return data, vocal2count
        
def preprocess(data, vocal2count, threshold):
    ''' replace the token word less then K with token 'Unk'
    '''
    uncommon_wordset = set()
    for word, count in vocal2count.items():
        if count <= threshold:
            uncommon_wordset.add(word)

    vocal = set(sorted(vocal2count.keys()))
    print('Vocabulary size: %d , %d words in uncommon word set' % (len(vocal), len(uncommon_wordset)))
    for sent in data:
        for idx, token in enumerate(sent.tokens):
            if token in uncommon_wordset:
                sent.tags[idx] = 'Unk'
            
    print('finish preprocessing')
    return vocal, data

def cal_trans_count(trn_data):
    ''' calculate transition count with shape = (K, K)
        K does not include Unk but including start
    '''
    trans_cnt = np.zeros((K, K))
    for sent in trn_data:
        sent_len = len(sent.tokens)
        for idx in range(sent_len+1):
            if idx == 0: # start
                prev_tag, curr_tag = 'START',  sent.tags[0]
                trans_cnt[tags2idx[prev_tag], tags2idx[curr_tag]] += 1
            elif idx == sent_len: # end
                prev_tag, curr_tag = sent.tags[-1], 'END'
                trans_cnt[tags2idx[prev_tag], tags2idx[curr_tag]] += 1
            elif idx >= 1 and idx <= sent_len-1: # middle
                prev_tag, curr_tag = sent.tags[idx-1], sent.tags[idx]
                trans_cnt[tags2idx[prev_tag], tags2idx[curr_tag]] += 1
            else:
                raise ValueError
    return trans_cnt

def cal_emit_count(trn_data, word2idx):
    ''' caculate emission count  with shape = (V, K)
    '''
    V = len(word2idx)
    emit_count = np.zeros((V, K))
    for sent in trn_data:
        sent_len = len(sent.tokens)
        for idx in range(sent_len):
            token, tag = sent.tokens[idx], sent.tags[idx]
            widx, tidx = word2idx[token], tags2idx[tag]
            emit_count[widx][tidx] += 1
    return emit_count

def save_trans_probs(trans_probs, fname):
    ''' save transmission probability with size (K, K) to txt file
    '''
    with open(fname, 'w') as f:
        for prev_tag_idx in range(trans_probs.shape[0]):
            for curr_tag_idx in range(trans_probs.shape[1]):
                prev_tag, curr_tag = idx2tags[prev_tag_idx], idx2tags[curr_tag_idx]
                trans_prob = trans_probs[prev_tag_idx][curr_tag_idx]
                line = ','.join([prev_tag, curr_tag, str(trans_prob)])+'\n'
                f.write(line)

def save_emit_probs(emit_probs, idx2word, fname):
    ''' save transmission probability with size (V, K) to txt file
    '''
    with open(fname, 'w') as f:
        for word_idx in range(emit_probs.shape[0]):
            for tag_idx in range(emit_probs.shape[1]):
                word, tag = idx2word[word_idx],  idx2tags[tag_idx]
                emit_prob = emit_probs[word_idx][tag_idx]
                line = ','.join([tag, word, str(emit_prob)])+'\n'
                f.write(line)

def load_dev_data(fname, vocal):
    ''' load dev data 
    '''
    with open(fname) as fin:
        raw_data = fin.read().splitlines()
    data = []
    for sent in raw_data:
        tokens, tags = [], []
        token_tag_pairs = sent.split(' ')
        for pair in token_tag_pairs:
            token, tag = pair.split('/')
            # token to lowercase
            token = token.lower()
            if token in vocal:
                tokens.append(token) 
                tags.append(tag)
            else:
                tokens.append(token)
                tags.append('Unk')
        sent = Sent(tokens, tags)
        data.append(sent)
    print("Loaded {} sentences in dev set".format(len(data)))
    return data

def viterbi_decoding(data, smooth_trans_logprobs, smooth_emit_logprobs, word2idx):
    ''' Viterbi_decoding algorithm
    '''
    for sent in data:
        pred_tags = []
        # dealing with the START token
        # calculate v0 with shape K 
        v_prev = smooth_trans_logprobs[tags2idx['START'], :]
        for idx, token in enumerate(sent.tokens):
            if token not in word2idx:
                pred_tags.append('Unk')
                continue
            if idx > 0:
                s = smooth_emit_logprobs[word2idx[token], :] + smooth_trans_logprobs[tags2idx[pred_tags[-1]], :]
            else:
                s = smooth_emit_logprobs[word2idx[token], :]
            v = s + v_prev
            max_idx = np.argmax(v)
            pred_tags.append(idx2tags[max_idx])
            v = v_prev
        sent.pred = pred_tags

def load_test_data(fname):
    ''' load test data 
    '''
    with open(fname) as fin:
        raw_data = fin.read().splitlines()
    data = []
    for sent in raw_data:
        tokens = sent.split(' ')
        # token to lowercase
        tokens = [token.lower() for token in tokens]
        sent = Sent(tokens=tokens)
        data.append(sent)
    print("Loaded {} sentences in test set".format(len(data)))
    return data

def test_acc(data):
    pred, label = [], []
    for sent in data:
        label.extend(sent.tags)
        pred.extend(sent.pred)
    return accuracy_score(label, pred)

def save_test_pos(tst_data, fname):
    with open(fname, 'w') as f:
        for sent in tst_data:
            tokens, preds = sent.tokens, sent.pred
            pairs = ['%s/%s'%(token, pred) for token, pred in zip(tokens, preds) ]
            line = ' '.join(pairs) + '\n'
            f.write(line)
    print('finish writing test set pos.')

def main():
    # load data
    trn_data, vocal2count = load_train_data('data/trn.pos')
    # preprocess data
    vocal, trn_data = preprocess(trn_data, vocal2count, threshold=1)
    # calculate transition count with shape = (K, K)
    trans_cnt = cal_trans_count(trn_data)
    # save to tprob.txt
    trans_tot_cnt = np.sum(trans_cnt, axis=1, keepdims=True)
    trans_tot_cnt[-1,:] = 1.0 # avoid divided by zero: no tags can follow END
    trans_probs = trans_cnt / trans_tot_cnt # last row is zero, first column is zero
    save_trans_probs(trans_probs, fname='data/jc6ub-tprob.txt')
    # smoothing and save smooth probablities
    smooth_trans_cnt = trans_cnt + BETA
    smooth_trans_tot_cnt = np.sum(smooth_trans_cnt, axis=1, keepdims=True)
    smooth_trans_probs = smooth_trans_cnt / smooth_trans_tot_cnt
    save_trans_probs(smooth_trans_probs, fname='data/jc6ub-tprob-smoothed.txt')
    # create word2idx and idx2word from vocal
    word2idx = {word:idx for idx, word in enumerate(vocal)}
    idx2word = {idx:word for word, idx in word2idx.items()}
    # caculate emission count shape = (V, K)
    emit_cnt = cal_emit_count(trn_data, word2idx)
    # save to eprob.txt
    emit_tot_cnt = np.sum(emit_cnt, axis=0, keepdims=True) # first and last rows are zero.
    emit_tot_cnt[:, 0]= emit_tot_cnt[:, -1] = 1.
    emit_probs = emit_cnt / emit_tot_cnt # first and last rows are zero.
    save_emit_probs(emit_probs, idx2word, fname='data/jc6ub-eprob.txt')
    # smoothing
    smooth_emit_cnt = emit_cnt + ALPHA
    smooth_emit_tot_cnt = np.sum(smooth_emit_cnt, axis=0, keepdims=True)
    smooth_emit_probs = smooth_emit_cnt / smooth_emit_tot_cnt
    save_emit_probs(smooth_emit_probs, idx2word, fname='data/jc6ub-eprob-smoothed.txt')
    # Viterbi algorithm in log space
    # convert probabilities to log space
    dev_data = load_dev_data('data/dev.pos', vocal)
    smooth_trans_logprobs, smooth_emit_logprobs = np.log(smooth_trans_probs), np.log(smooth_emit_probs)
    viterbi_decoding(dev_data, smooth_trans_logprobs, smooth_emit_logprobs, word2idx)
    dev_acc = test_acc(dev_data)
    print('using Viterbi decoding, dev acc:', dev_acc)
    # test on test set
    tst_data = load_test_data('data/tst.word')
    viterbi_decoding(tst_data, smooth_trans_logprobs, smooth_emit_logprobs, word2idx)
    save_test_pos(tst_data, fname='data/jc6ub-viterbi.txt')

if __name__ == '__main__':
    main()