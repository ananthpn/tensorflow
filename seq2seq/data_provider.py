# Copyright 2017 Anantharaman Palacode Narayana Iyer. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import random
import string
import numpy as np

class DataProvider(object):
    """"Provides a vector representation for the given ASCII string"""
    def __init__(self, num_samples, max_str_len=32, alphabet=None, batch_size=32):
        self.num_samples = num_samples
        self.max_str_len = max_str_len # we will generate strings in the range 1 char to max_str_len chars
        self.chars = alphabet
        self.batch_size = batch_size

        self.batch_num = 0 # batch number that the iterator is tracking
        self.inputs, self.seqlen, self.d_inputs, self.d_seqlen, self.targets_e, self.targets_d = self.generate_ds()
        self.pad_value = 0 # we will do 0 padding after seqlen
        self.max_seq_len = 32
        return
    
    def get_embeddings(self):
        # define the embedding matrix to map decoder symbols to a vector representation
        num_symbols = 128 # total number of distinct symbols of decoder
        embeddings = []
        for symbol in range(num_symbols):
            vec = [0] * num_symbols
            vec[symbol] = 1
            embeddings.append(vec)
        return embeddings
    
    def next(self):
        # maxlen is determined by the max seq len found in the batch
        get_batch = lambda my_list, start, size: my_list[(start * size) :(start*size) + size] 
        inputs =  get_batch(self.inputs, self.batch_num, self.batch_size)
        seqlen =  get_batch(self.seqlen, self.batch_num, self.batch_size)
        d_inputs =  get_batch(self.d_inputs, self.batch_num, self.batch_size)
        d_seqlen =  get_batch(self.d_seqlen, self.batch_num, self.batch_size)
        targets_e =  get_batch(self.targets_e, self.batch_num, self.batch_size)
        targets_d =  get_batch(self.targets_d, self.batch_num, self.batch_size)
        self.batch_num += 1
        
        maxlen = self.get_batch_maxlen(targets_d) # get the max seqlen of the given batch
        
        for x, xd, target in zip(inputs, d_inputs, targets_d):
            diff = maxlen - len(target) # amount of padding needed
            for _ in range(diff):
                vec = [0] * 128
                vec[self.pad_value] = 1
                target.append(vec)
                xd.append(vec)
                x.append(vec)
        assert seqlen == d_seqlen, "In our dataset seqlen and d_seqlen should b same"
        return inputs, seqlen, d_inputs, d_seqlen, targets_e, targets_d
    
    def get_batch_maxlen(self, targets_d):
        mlen = 0
        for seq in targets_d:
            #print "len seq: ", len(seq)
            if len(seq) > mlen:
                mlen = len(seq)
        return mlen
    
    def reset(self):
        self.batch_num = 0
        return

    def generate_random_strings(self):
        inputs = []
        if self.chars == None:
            self.chars = [chr(x) for x in range(ord("0"), ord("9"))] #range("0", "9") #string.printable
        for _sample in range(self.num_samples):
            wlen = random.randint(1, self.max_str_len - 1)
            word = ""
            for _ in range(wlen):
                word += random.choice(self.chars)
            inputs.append(word)
        return inputs
    
    def generate_ds(self):
        input_strs = self.generate_random_strings()
        e_inputs = []
        e_targets = [] # targets for the encoder
        d_inputs = []
        d_targets = []
        seqlen = []
        seqlen_d = []
        eos_vec = [0] * 128 #
        eos_vec[0] = 1 # we use 0 as the terminator 
        
        for word in input_strs: # for each word in the given input list
            w_input = [] # word input is a seq of char reps
            if len(word) == 0:
                print "got a 0 len word: ", word
            for c in word: # for each char
                vec = [0] * 128 # create a 1-h vec for Softmax trg
                vec[ord(c)] = 1
                w_input.append(vec)

            e_inputs.append(w_input[:])
            e_targets.append([w_input[0]]) # the first char will be the encoder's output
            seqlen.append(len(w_input))
            assert len(w_input) >= 1, "w_input should have a len greater than eq 1, got %d" % (len(w_input))
            d_inputs.append(w_input[:])
            tgt = w_input[1:]
            tgt.append(eos_vec)
            d_targets.append(tgt)            
            seqlen_d.append(len(w_input))
            
        assert len(e_inputs) == len(d_inputs), "number of inputs to encoder %d should equal %d decoder" % (len(e_inputs), len(d_inputs))
        assert len(e_inputs) == len(e_targets), "number of inputs to encoder %d should equal %d encoder targets" % (len(e_inputs), len(e_targets))
        assert len(seqlen) == len(seqlen_d), "seqlen %d should equal %d decoder seqlen for our dataset" % (len(e_inputs), len(d_inputs))
        return e_inputs, seqlen, d_inputs, seqlen_d, e_targets, d_targets
    
if __name__ == "__main__":
    sr = DataProvider(16, 10) # num samples, max len
    e_inputs, seqlen, d_inputs, d_seqlen, d_targets = sr.next()
    print d_inputs[:1]    
    print len(e_inputs[0]), len(e_inputs[0][0]), len(d_inputs[0]), len(d_inputs[0][0]), 