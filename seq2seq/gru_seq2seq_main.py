import os
from gru_seq2seq import GRU_Seq2Seq
from data_provider import DataProvider
import numpy as np

def get_model(tr):
    clf_pic_name = os.path.join(".", "tf_str_seq2seq_sm.sess") # this will be used to save the model   
    num_sequences = 15 * 1024
    batch_size = 16
    dgen = DataProvider(num_sequences, mlen, batch_size=batch_size )
    input_dims = 128
    hidden_size = 256 #128
    num_decoder_symbols = 128
    num_epochs = 10
    
    # clf_pic_name, input_dims, hidden_size, num_decoder_symbols,
    s2s = GRU_Seq2Seq(clf_pic_name, input_dims, hidden_size, num_decoder_symbols)
    if tr == "YES":
        s2s.train(dgen, num_epochs)
    return s2s

def validate(s2s, n_samples):
    bsize = 1 # batch size        
    vgen = DataProvider(n_samples, mlen, batch_size=bsize )

    correct = 0
    correct_elements = 0
    total = 0
    total_elements = 0

    for _ in range(n_samples):
        batch, slen, _d_inputs, _d_seqlen, _targets_e, _targets_d = vgen.next()
        inp = []
        for i, b1 in enumerate(batch):
            for j in range(slen[i]):
                inp.append(chr(b1[j].index(1)))
        e_results, results = s2s.do_inference(batch, slen, vgen)
        pred = [chr(np.argmax(e_results))]
        for i, result in enumerate(results):
            for j, res in enumerate(result): # for each seq in a mini batch
                pred.append(chr(np.argmax(res)))
        pred = pred[:-1] # ignore the end char
        print "Inp: ", inp
        print "Prd: ", pred
        for c1, c2 in zip(inp, pred):
            if c1 == c2:
                correct_elements += 1
            total_elements += 1
        if inp == pred:
            correct += 1
        total += 1
    
    print "EXACT match validation accuracy: ", (float(correct)/total) * 100, "%"
    print "Elementwise match validation accuracy: ", (float(correct_elements)/total_elements) * 100, "%"

if __name__ == '__main__':
    mlen = 16 # max length of the string to produce
    num_validation_sequences = 100
    tr = raw_input("Should I train? (YES/*) ")
    s2s = get_model(tr)
    validate(s2s, num_validation_sequences)