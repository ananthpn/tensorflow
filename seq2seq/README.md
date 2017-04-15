# seq2seq: TensorFlow based Encoder-Decoder
This package implements the sequence to sequence model as described in the paper: Sequence to Sequence Learning with Neural Networks by Ilya Sutskever, Oriol Vinyals, Quoc V. Le. This is also referred to as Encoder-Decoder architecture.  

<b>Important Note</b>:  This package implemented using TensorFlow v 1.0. As the APIs change rapidly with TensorFlow versions, please ensure that you are on TensorFlow Version 1.0 when using this package.

Steps for Usage:
----------------
- Create a data provider class. The file data_provider.py has a sample. It is necessary to support the interface methods next(), reset(), get_embeddings(). This class is an iterator that is required to provide a mini-batch of samples every time next() is called. When the iterator reaches the end of dataset it should return empty lists. The return values of next() should be: e_inputs, e_seqlen, d_inputs, d_seqlen, e_targets, d_targets.

e_inputs: Inputs to the encoder with shape: (mini_batch_size, max_time_step_in_the_mini_batch, input_vector_dimension). For example, if you are returning a mini batch 16 sequences, max time step across all 16 sequences is 10 and the input vector is 128 dimensional, e_inputs will have the shape: (16, 10, 128). Each sequence should be padded as needed to be of the fixed length equal to max_time_step_in_the_mini_batch, while the sequence length parameter (discussed below) is a list that contains the exact length of each sequence. Padding is done more to make the linear algebric computations more efficient.
e_seqlen: This is a list of sequence lengths of each sequence in the mini batch. For example, if the mini batch size is 4 sequences and each sequence has a length 7, 4, 10, 6 respectively, this will be: [7, 4, 10, 6]. But note that e_inputs for this mini batch will have sequences that have a fixed length of 10 (as in this example, 10 is the max_time_step_in_the_mini_batch) where the first sequence is padded with 3 additional vectors, second sequence with 6 and so on. One may choose the pad vector to be a vector of zeros for example.
d_inputs: Same as e_inputs but for decoder
d_seqlen: Same as e_seqlen but for decoder
e_targets: Targets for the encoder GRU. This will have the shape: (batch_size, 1, number_of_output_units_of_encoder). Our current implementation is based on Softmax output layer and hence the target output is a one hot vector.
d_targets: Targets for the decoder. This is similar to e_targets but there will be a one hot vector for every decoder time step.
Please see the file data_provider.py for details.
- Create an instance of GRU_seq2seq class
- Train the seq2seq
- Do inferencing

Usage Example
------
clf_pic_name = os.path.join(".", "tf_str_seq2seq_sm.sess") # this will be used to save the model   

num_sequences = 15 * 1024

batch_size = 16

dgen = DataProvider(num_sequences, mlen, batch_size=batch_size )
input_dims = 128
hidden_size = 256 #128
num_decoder_symbols = 128
num_epochs = 10


#create an instance of seq2seq with clf_pic_name, input_dims, hidden_size, num_decoder_symbols

s2s = GRU_Seq2Seq(clf_pic_name, input_dims, hidden_size, num_decoder_symbols)

#Train the model

s2s.train(dgen, num_epochs)

#Do inferencing

vgen = DataProvider(num_samples_for_validation, mlen, batch_size=bsize ) # vgen is the data provider for validation

#e_results are results of the encoder and results are the results from decoder inferencing

e_results, results = s2s.do_inference(batch, slen, vgen)
