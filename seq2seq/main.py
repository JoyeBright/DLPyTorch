from modules.lib import *

# read the input data
d = DataReader("eng", "fra", config_WOAttention['max_length_data'], True)
input_lang, output_lang, pairs = d.PrepareData()
# prints a random pair
print(random.choice(pairs))
# initialize the encoder
encoder1 = EncoderRNN(input_lang.n_words, config_WOAttention['hidden_size']).to(config_WOAttention['device'])
print(encoder1)
# initialize the decoder
decoder1 = DecoderRNN(config_WOAttention['hidden_size'], output_lang.n_words).to(config_WOAttention['device'])
print(decoder1)

trainIters(encoder1, decoder1, pairs, input_lang, output_lang, config_WOAttention['n_iters'],\
           config_WOAttention['print_every'], config_WOAttention['plot_every'],\
           config_WOAttention['learning_rate'], config_WOAttention['device']
           )

print("Encoder's state_dict:")
for param_tensor in encoder1.state_dict():
    print(param_tensor, "\t", encoder1.state_dict()[param_tensor].size())

print("Decoder's state_dict:")
for param_tensor in decoder1.state_dict():
    print(param_tensor, "\t", decoder1.state_dict()[param_tensor].size())
