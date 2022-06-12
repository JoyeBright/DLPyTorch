import time
import torch
import torch.nn as nn
from torch import optim
from models.config_WOAttention import *
from helpers.DataReader import tensorsFromPair
from helpers.Timer import timeSince
from helpers.Plot import showPlot

class EncoderRNN(nn.Module):
    """
    Given an input size which is the total number of words in the input/src dictionary
    and hidden_layer size which is a pre-defined number (can be set through the config file),
    EncoderRNN creates a RNN encoder with an embedding layer followed by a GRU layer.
    embedding layer (num_embeddings, embedding_dim)
    GRU layer (input_size, hidden_size)
    Forward_pass:
    (i) feed input data to the embedding layer and name it "output"
    (ii) pass the "output" and "hidden" data to the GRU layer
    step(ii) generates output and final hidden state for the input sequence
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size) #num_embeddings, embedding_dim
        self.gru = nn.GRU(hidden_size, hidden_size) #input_size (x), hidden_size (h)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden =  self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device = config_WOAttention['device'])

class DecoderRNN(nn.Module):
    """
    Given a hidden_layer size which is a pre-defined number (can be set through the config file),
    and an input size which is the total number of words in the output/trg dictionary, DecoderRNN
    creates a RNN decoder with an embedding layer followed by a GRU layer, a single layer feed-Forward,
    and an activation function, LogSoftmax.
    embedding layer (num_embeddings, embedding_dim)
    GRU layer (input_size, hidden_size)
    linear layer (hidden_size, output_size)
    Forward_pass:
    (i) feed input data layer to the embedding layer and name it output
    (ii) pass the "output" and "hidden" data to the GRU layer
    (iii) change the dimensionality of the output via a fully connected layer
    (iv) add a softmax AF on top of that
    """
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        linear = self.out(output[0])
        output = self.softmax(linear)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config_WOAttention['device'])

def train(input_tensor, target_tensor, encoder, decoder, \
             encoder_optimizer, decoder_optimizer, criterion, max_length = config_WOAttention['max_length']):

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad() #Sets the gradients of all optimized torch.Tensor s to zero
    decoder_optimizer.zero_grad() #Sets the gradients of all optimized torch.Tensor s to zero

    input_length  = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=config_WOAttention['device'])

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0] # [1,1, hidden_size = if 256] = 256

    decoder_input = torch.tensor([[config_WOAttention['SOS_token']]], device=config_WOAttention['device'])

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == config_WOAttention['EOS_token']:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, pairs, input_lang, output_lang, n_iters,\
               print_every, plot_every, learning_rate, device):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # print("encoder_optimizer", encoder_optimizer)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # print("decoder_optimizer", decoder_optimizer)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang, device) for i in range(n_iters)]

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        # print("iter %.f\n" % iter)
        training_pair = training_pairs[iter - 1]
        # print("training_pair", training_pair)
        input_tensor = training_pair[0]
        # print("input_tensor", input_tensor)
        target_tensor = training_pair[1]
        # print("target_tensor", target_tensor)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        # print(loss)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
