from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score

K = 12
tags2idx = {'START':0, 'A':1, 'C':2, 'D':3, 'M':4, 'N':5,\
                'O':6, 'P':7, 'R':8, 'V':9, 'W':10, 'END':11}
idx2tags = {idx: tag for tag, idx in tags2idx.items()}
print(idx2tags)
ALPHA = 1
BETA = 1

class Sent(object):
    def __init__(self, tokens=None, tags=None):
        self.tokens = tokens
        self.tags = tags
        self.preds = None

def load_train_data(fname):
    ''' load train data 
    '''
    with open(fname) as fin:
        raw_data = fin.read().splitlines()
    data = []
    token2count = Counter()
    for sent in raw_data:
        tokens, tags = [], []
        token_tag_pairs = sent.split(' ')
        for pair in token_tag_pairs:
            token, tag = pair.split('/')
            # token to lowercase
            token = token.lower()
            tokens.append(token), tags.append(tag)
        token2count.update(tokens)
        sent = Sent(tokens, tags)
        data.append(sent)
    print("Loaded {} sentences in train set".format(len(data)))
    return data, token2count
        
def preprocess(data, token2count, threshold):
    ''' replace the token word less then K with token 'Unk'
    '''
    uncommon_wordset = set()
    vocal = set()
    for token, count in token2count.items():
        if count <= threshold:
            uncommon_wordset.add(token)
        else:
            vocal.add(token)
    for sent in data:
        for idx, token in enumerate(sent.tokens):
            if token in uncommon_wordset:
                sent.tokens[idx] = 'Unk'
    vocal.add('Unk')
    print('Vocabulary size: %d , %d words in uncommon word set' % (len(vocal), len(uncommon_wordset)))          
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
    ''' caculate emission count with shape = (V, K)
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
                if prev_tag == 'END' or curr_tag == 'START':
                    continue
                trans_prob = trans_probs[prev_tag_idx][curr_tag_idx]
                line = ','.join([prev_tag, curr_tag, str(trans_prob)])+'\n'
                f.write(line)

def save_emit_probs(emit_probs, idx2word, fname):
    ''' save transmission probability with size (V, K) to txt file
    '''
    with open(fname, 'w') as f:
        for word_idx in range(emit_probs.shape[0]):
            for tag_idx in range(emit_probs.shape[1]):
                if idx2tags[tag_idx] == 'END' or idx2tags[tag_idx] == 'START':
                    continue
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
            tags.append(tag)
            tokens.append(token) # if token in vocal else tokens.append('Unk')
        sent = Sent(tokens, tags)
        data.append(sent)
    print("Loaded {} sentences in dev set".format(len(data)))
    return data

def viterbi_decoding(data, smooth_trans_logprobs, smooth_emit_logprobs, word2idx):
    ''' Viterbi_decoding algorithm
    '''
    for sent in data:
        # intialization
        pred_tags = []
        sent_len = len(sent.tokens)
        V = np.zeros((sent_len, K)) # v shape = (sent_len, k)
        B = np.zeros((sent_len, K), dtype=int) # b shape = (sent_len, k) or (sent_len-1, k)
        # v1
        token = sent.tokens[0] if sent.tokens[0] in word2idx else 'Unk'
        V[0,:] = smooth_trans_logprobs[tags2idx['START'], :] + smooth_emit_logprobs[word2idx[token], :]
        B[0,:] = 0
        # forward
        for idx in range(1, sent_len):
            token = sent.tokens[idx] if sent.tokens[idx] in word2idx else 'Unk'
            for k in range(K):
                scores = smooth_trans_logprobs[:,k] + smooth_emit_logprobs[word2idx[token],k] + V[idx-1, :]
                V[idx, k] = max(scores)
                B[idx, k] = np.argmax(scores)
        scores = smooth_trans_logprobs[:,tags2idx['END']] + V[-1, :]
        y = np.argmax(scores)
        pred_tags.append(idx2tags[y])
        # backward
        for idx in range(sent_len-1, 0, -1):
            pred_tags.insert(0, idx2tags[B[idx, y]])
            y = B[idx, y]
        sent.preds = pred_tags

def test_acc(data):
    preds, labels = [], []
    for sent in data:
        labels.extend(sent.tags)
        preds.extend(sent.preds)
    return accuracy_score(labels, preds)

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

def save_test_pos(tst_data, fname):
    with open(fname, 'w') as f:
        for sent in tst_data:
            tokens, preds = sent.tokens, sent.preds
            pairs = ['%s/%s'%(token, pred) for token, pred in zip(tokens, preds) ]
            line = ' '.join(pairs) + '\n'
            f.write(line)
    print('finish writing test set pos.')

def main():
    # load data
    trn_data, token2count = load_train_data('data/trn.pos')
    # preprocess data
    vocal, trn_data = preprocess(trn_data, token2count, threshold=1)
    # calculate transition count with shape = (K, K)
    trans_cnt = cal_trans_count(trn_data)
    # save to tprob.txt
    trans_tot_cnt = np.sum(trans_cnt, axis=1, keepdims=True)
    trans_tot_cnt[-1,:] = 1.0 # avoid divided by zero: no tags can follow END
    trans_probs = trans_cnt / trans_tot_cnt # last row is zero, first column is zero
    save_trans_probs(trans_probs, fname='data/jc6ub-tprob.txt')
    # smoothing and save smooth probablities
    smooth_trans_cnt = trans_cnt + BETA
    smooth_trans_cnt[:,0] = smooth_trans_cnt[-1,:] = 0. # make smooth value correct
    smooth_trans_tot_cnt = np.sum(smooth_trans_cnt, axis=1, keepdims=True)
    smooth_trans_tot_cnt[-1,:] = 1.0 # avoid divided by zero: no tags can follow END
    smooth_trans_probs = smooth_trans_cnt / smooth_trans_tot_cnt
    save_trans_probs(smooth_trans_probs, fname='data/jc6ub-tprob-smoothed.txt')
    # create word2idx and idx2word from vocal
    word2idx = {word:idx for idx, word in enumerate(vocal)}
    idx2word = {idx:word for word, idx in word2idx.items()}
    # caculate emission count shape = (V, K)
    emit_cnt = cal_emit_count(trn_data, word2idx)
    # save to eprob.txt
    emit_tot_cnt = np.sum(emit_cnt, axis=0, keepdims=True) # first and last rows are zero.
    emit_tot_cnt[:, 0] = emit_tot_cnt[:, -1] = 1.
    emit_probs = emit_cnt / emit_tot_cnt # first and last rows are zero.
    save_emit_probs(emit_probs, idx2word, fname='data/jc6ub-eprob.txt')
    # smoothing
    smooth_emit_cnt = emit_cnt + ALPHA
    smooth_emit_cnt[:,0] = smooth_emit_cnt[:, -1] = 0
    smooth_emit_tot_cnt = np.sum(smooth_emit_cnt, axis=0, keepdims=True)
    smooth_emit_tot_cnt[:, 0] = smooth_emit_tot_cnt[:, -1] = 1.
    smooth_emit_probs = smooth_emit_cnt / smooth_emit_tot_cnt
    save_emit_probs(smooth_emit_probs, idx2word, fname='data/jc6ub-eprob-smoothed.txt')
    # convert probabilities to log space
    dev_data = load_dev_data('data/dev.pos', vocal)
    smooth_trans_probs[:,0] = smooth_trans_probs[-1,:] = 1e-200 # avoid dividing by zero encountered in log
    smooth_emit_probs[:,0] = smooth_emit_probs[:,-1] = 1e-200 # avoid dividing by zero encountered in log
    smooth_trans_logprobs, smooth_emit_logprobs = np.log(smooth_trans_probs), np.log(smooth_emit_probs)
    # Viterbi algorithm in log space
    viterbi_decoding(dev_data, smooth_trans_logprobs, smooth_emit_logprobs, word2idx)
    dev_acc = test_acc(dev_data)
    print('using Viterbi decoding, dev acc:', dev_acc)
    # test on test set
    tst_data = load_test_data('data/tst.word')
    viterbi_decoding(tst_data, smooth_trans_logprobs, smooth_emit_logprobs, word2idx)
    save_test_pos(tst_data, fname='data/jc6ub-viterbi.txt')

if __name__ == '__main__':
    main()