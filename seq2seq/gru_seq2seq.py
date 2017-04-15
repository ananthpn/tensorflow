"""
Created on April 09, 2017
@author: Anantharaman Narayana Iyer
Encoder Decoder (Seq2seq) implementation using TF and a sample progression application
that builds on dynamic_rnn example
We will use logistic layer in this example
We use tf seq2seq API in this implementation - version of tf API r 1.0
See: https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder
"""
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

class GRU_Seq2Seq():
    def __init__(self, clf_pic_name, input_dims, hidden_size, num_decoder_symbols, learning_rate=0.01, maxlen_to_decode=16):
        self.clf_pic_name = clf_pic_name # we will save the model here
        # set up some common variables
        self.start_of_sequence_id = 0 # this will help us to terminate the seq
        self.end_of_sequence_id = 0
        self.encoder_hidden_size = hidden_size
        self.decoder_hidden_size = self.encoder_hidden_size
        self.learning_rate = learning_rate
        self.decoder_sequence_length = maxlen_to_decode #7 #max length that decoder will predict before terminating

        # placeholders and variables
        self.encoder_length = tf.placeholder(tf.int32, [None]) # seq length for dynamic time unrolling
        self.decoder_length = tf.placeholder(tf.int32, [None]) # seq length for dynamic time unrolling
        self.encoder_embedding_size = input_dims #
        self.decoder_embedding_size = self.encoder_embedding_size
        self.decoder_embeddings = tf.get_variable('decoder_embeddings',
                [self.decoder_embedding_size, self.decoder_embedding_size],) # 
        self.num_decoder_symbols = num_decoder_symbols #self.decoder_embedding_size # number of output classes of decoder
        with tf.variable_scope("rnn") as scope:
            # setting up weights for computing the final output
            self.output_fn = lambda x: layers.linear(x, self.num_decoder_symbols,
                                          scope=scope)
        self.inputs = tf.placeholder("float", [None, None, self.encoder_embedding_size])
        self.decoder_inputs = tf.placeholder("float", [None, None, self.decoder_embedding_size])
        self.encoder_targets = tf.placeholder("float", [None, None, self.num_decoder_symbols])
        self.decoder_targets = tf.placeholder("float", [None, None, self.num_decoder_symbols])
                
        # build model - compute graph
        self.encoder()
        self.decoder_train()
        self.decoder_inference()
        self.compute_cost()
        self.optimize()
        self.get_sm_outputs()
        return
    
    def encoder(self):
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                        cell=tf.contrib.rnn.GRUCell(self.encoder_hidden_size), inputs=self.inputs, sequence_length=self.encoder_length, 
                        dtype=tf.float32, scope="rnn", time_major=False) # tf.variable_scope("rnn")
        self.encoder_outputs = tf.contrib.layers.linear(self.encoder_state, self.num_decoder_symbols)
        self.encoder_softmax_outputs = tf.nn.softmax(self.encoder_outputs)
        return #encoder_outputs, encoder_state

    def decoder_train(self):
        with tf.variable_scope("decoder") as scope:
            # Train decoder
            self.decoder_cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_size)
            decoder_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(
                        encoder_state=self.encoder_state)
            self.decoder_outputs_train, self.decoder_state_train, _ = (
                        tf.contrib.seq2seq.dynamic_rnn_decoder(
                                    cell=self.decoder_cell,
                                    decoder_fn=decoder_fn_train,
                                    inputs=self.decoder_inputs,
                                    sequence_length=self.decoder_length,
                                    time_major=False,
                                    scope=scope))
            self.decoder_outputs_train = self.output_fn(self.decoder_outputs_train)
        return #decoder_outputs_train, decoder_state_train, decoder_cell
        
    def decoder_inference(self):
        # Inference decoder
        with tf.variable_scope("decoder") as scope:
            # Setup variable reuse
            scope.reuse_variables()
            decoder_fn_inference = (
                tf.contrib.seq2seq.simple_decoder_fn_inference(
                output_fn=self.output_fn,
                encoder_state=self.encoder_state,
                embeddings=self.decoder_embeddings,
                start_of_sequence_id= self.start_of_sequence_id,
                end_of_sequence_id = self.end_of_sequence_id,
                maximum_length=self.decoder_sequence_length-1,
                num_decoder_symbols=self.num_decoder_symbols,
                dtype=tf.int32))
            
            self.decoder_outputs_inference, self.decoder_state_inference, _ = (
                tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=self.decoder_cell,
                decoder_fn=decoder_fn_inference,
                #sequence_length=decoder_length,
                time_major=False,
                scope=scope))
        return # decoder_outputs_inference, decoder_state_inference
    
    def compute_cost(self):
        self.e_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.encoder_outputs, labels=self.encoder_targets))
        self.d_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.decoder_outputs_train, labels=self.decoder_targets))
        self.cost = self.e_cost + self.d_cost
        return self.cost
    
    def optimize(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        return

    def get_sm_outputs(self):
        self.sm_outputs = tf.nn.softmax(self.decoder_outputs_inference)
        return self.sm_outputs
    
    def train(self, dgen, num_epochs=7):
        saver = tf.train.Saver() 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print "After global init"
            
            for ep in range(num_epochs):
                ep_cost = 0.0 # cost per epoch
                while True:
                    batch_X, slen, batch_XD, d_slen, batch_Y, batch_YD = dgen.next()
                    if len(batch_X) == 0:
                        break
                    _decoder_outputs_train_res, _decoder_state_train_res = sess.run(
                        [self.decoder_outputs_train, self.decoder_state_train], 
                        feed_dict= {self.inputs: batch_X, self.decoder_inputs: batch_XD, self.encoder_length: slen, self.decoder_length: d_slen})
                                    
                    _, _cost = sess.run([self.optimizer, self.cost], 
                        feed_dict= {self.inputs: batch_X, self.decoder_inputs: batch_XD, self.encoder_length: slen, self.decoder_length: d_slen, 
                                    self.encoder_targets: batch_Y, self.decoder_targets: batch_YD})
                    ep_cost += _cost
                print "For epoch %d, cost = %f" % (ep, ep_cost)
                dgen.reset()        
            print "After train, going to save the model at: ", self.clf_pic_name
            _fn = saver.save(sess, self.clf_pic_name)
        return

    def do_inference(self, batch_X, slen, dgen):
        saver = tf.train.Saver() # restore the saved session 
        with tf.Session() as sess:
            saver.restore(sess, self.clf_pic_name) # restore the saved session
            embs = dgen.get_embeddings()
            e_state = sess.run([self.encoder_state], feed_dict = {self.inputs: batch_X, self.encoder_length: slen})
            e_state = np.reshape(e_state, [-1, self.decoder_hidden_size])
            e_results, d_results = sess.run([self.encoder_softmax_outputs, self.sm_outputs], feed_dict = {self.encoder_state:e_state, self.decoder_embeddings:embs})
        return e_results, d_results #decoder_outputs_inference_res
