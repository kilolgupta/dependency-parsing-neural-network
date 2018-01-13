import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dynet as dynet
import random
import os
import numpy as np
from decoder import *
from utils import *

class DepModel:
    def __init__(self):

        # first initialize a computation graph container (or model).
        self.nnmodel = dynet.Model()

        # assign the algorithm for backpropagation updates.
        self.updater = dynet.AdamTrainer(self.nnmodel)

        num_words, num_tags, num_labels = 4808, 46, 46
        word_embed_dim, pos_embed_dim, label_embed_dim = 100, 32, 32
        hidden_layer1_dim, hidden_layer2_dim = 600, 600
        num_actions = 93

        self.minibatch_size = 1000

        # create embeddings for words and tag features.
        self.word_embedding = self.nnmodel.add_lookup_parameters((num_words, word_embed_dim))

        glove_word_embeddings_dict = {} # key is the word, value is the list of 100 embeddings
        embed_lines = open("glove.6B.100d.txt", 'r').read().splitlines()
        for line in embed_lines:
            word = line.split()[0]
            values = line.split()
            del values[0]
            glove_word_embeddings_dict[word] = values

        vocab_words = open("./data/vocabs.word", 'r').read().splitlines()
        i = 0
        for word_line in vocab_words:
            word = word_line.split()[0]
            if(word in glove_word_embeddings_dict):
                self.word_embedding[i] = np.asarray(glove_word_embeddings_dict[word])
                
        self.pos_embedding = self.nnmodel.add_lookup_parameters((num_tags, pos_embed_dim))
        self.label_embedding = self.nnmodel.add_lookup_parameters((num_labels, label_embed_dim))

        # mbda x: dynet.bmax(.1 * x, x))assign transfer function
        self.transfer = (lambda x: dynet.bmax(.1 * x, x))

        self.input_dim = 20*(word_embed_dim + pos_embed_dim) + 12*label_embed_dim

        self.hidden_layer1 = self.nnmodel.add_parameters((hidden_layer1_dim, self.input_dim))
        self.hidden_layer1_bias = self.nnmodel.add_parameters(hidden_layer1_dim, init=dynet.ConstInitializer(0.2))

        self.hidden_layer2 = self.nnmodel.add_parameters((hidden_layer2_dim, hidden_layer1_dim))
        self.hidden_layer2_bias = self.nnmodel.add_parameters(hidden_layer2_dim, init=dynet.ConstInitializer(0.2))

        # define the output weight.
        self.output_layer = self.nnmodel.add_parameters((num_actions, hidden_layer2_dim))

        # define the bias vector and initialize it as zero.
        self.output_bias = self.nnmodel.add_parameters(num_actions, init=dynet.ConstInitializer(0))

        self.dropout_prob = 0.2
        '''
            You can add more arguments for examples actions and model paths.
            You need to load your model here.
            actions: provides indices for actions.
            it has the same order as the data/vocabs.actions file.
        '''
        # if you prefer to have your own index for actions, change this.
        self.actions = ['SHIFT', 'LEFT-ARC:prep', 'LEFT-ARC:dobj', 'LEFT-ARC:poss', 'LEFT-ARC:amod', 'LEFT-ARC:xcomp', 'LEFT-ARC:mark', 'LEFT-ARC:conj', 'LEFT-ARC:nn', 'LEFT-ARC:rcmod', 'LEFT-ARC:advcl', 'LEFT-ARC:cc', 'LEFT-ARC:pcomp', 'LEFT-ARC:expl', 'LEFT-ARC:tmod', 'LEFT-ARC:csubj', 'LEFT-ARC:number', 'LEFT-ARC:iobj', 'LEFT-ARC:<null>', 'LEFT-ARC:preconj', 'LEFT-ARC:nsubj', 'LEFT-ARC:appos', 'LEFT-ARC:infmod', 'LEFT-ARC:partmod', 'LEFT-ARC:ccomp', 'LEFT-ARC:aux', 'LEFT-ARC:auxpass', 'LEFT-ARC:parataxis', 'LEFT-ARC:det', 'LEFT-ARC:punct', 'LEFT-ARC:discourse', 'LEFT-ARC:dep', 'LEFT-ARC:cop', 'LEFT-ARC:pobj', 'LEFT-ARC:num', 'LEFT-ARC:prt', 'LEFT-ARC:possessive', 'LEFT-ARC:rroot', 'LEFT-ARC:npadvmod', 'LEFT-ARC:mwe', 'LEFT-ARC:neg', 'LEFT-ARC:predet', 'LEFT-ARC:nsubjpass', 'LEFT-ARC:quantmod', 'LEFT-ARC:root', 'LEFT-ARC:acomp', 'LEFT-ARC:advmod', 'RIGHT-ARC:prep', 'RIGHT-ARC:dobj', 'RIGHT-ARC:poss', 'RIGHT-ARC:amod', 'RIGHT-ARC:xcomp', 'RIGHT-ARC:mark', 'RIGHT-ARC:conj', 'RIGHT-ARC:nn', 'RIGHT-ARC:rcmod', 'RIGHT-ARC:advcl', 'RIGHT-ARC:cc', 'RIGHT-ARC:pcomp', 'RIGHT-ARC:expl', 'RIGHT-ARC:tmod', 'RIGHT-ARC:csubj', 'RIGHT-ARC:number', 'RIGHT-ARC:iobj', 'RIGHT-ARC:<null>', 'RIGHT-ARC:preconj', 'RIGHT-ARC:nsubj', 'RIGHT-ARC:appos', 'RIGHT-ARC:infmod', 'RIGHT-ARC:partmod', 'RIGHT-ARC:ccomp', 'RIGHT-ARC:aux', 'RIGHT-ARC:auxpass', 'RIGHT-ARC:parataxis', 'RIGHT-ARC:det', 'RIGHT-ARC:punct', 'RIGHT-ARC:discourse', 'RIGHT-ARC:dep', 'RIGHT-ARC:cop', 'RIGHT-ARC:pobj', 'RIGHT-ARC:num', 'RIGHT-ARC:prt', 'RIGHT-ARC:possessive', 'RIGHT-ARC:rroot', 'RIGHT-ARC:npadvmod', 'RIGHT-ARC:mwe', 'RIGHT-ARC:neg', 'RIGHT-ARC:predet', 'RIGHT-ARC:nsubjpass', 'RIGHT-ARC:quantmod', 'RIGHT-ARC:root', 'RIGHT-ARC:acomp', 'RIGHT-ARC:advmod']
        # write your code here for additional parameters.
        # feel free to add more arguments to the initializer.

    def forward(self, string_features, is_train):

        # extract word, tags and label ids
        word_ids = [word2id(word_feat) for word_feat in string_features[0:20]]
        tag_ids = [tag2id(tag_feat) for tag_feat in string_features[20:40]]
        label_ids = [label2id(label_feat) for label_feat in string_features[40:52]]
	
        # extract word embeddings and tag embeddings from features
        word_embeds = [self.word_embedding[wid] for wid in word_ids]
        tag_embeds = [self.pos_embedding[tid] for tid in tag_ids]
        label_embeds = [self.label_embedding[lid] for lid in label_ids]

        # concatenating all features (recall that '+' for lists is equivalent to appending two lists)
        embedding_layer = dynet.concatenate(word_embeds + tag_embeds + label_embeds)

        # calculating the hidden layer
        # .expr() converts a parameter to a matrix expression in dynetnet (its a dynetnet-specific syntax).
        hidden1 = self.transfer(self.hidden_layer1.expr() * embedding_layer + self.hidden_layer1_bias.expr())

        if(is_train):
            dynet.dropout(hidden1, self.dropout_prob)

        hidden2 = self.transfer(self.hidden_layer2.expr() * hidden1 + self.hidden_layer2_bias.expr())

        if(is_train):
            dynet.dropout(hidden2, self.dropout_prob)

        # calculating the output layer
        output = self.output_layer.expr() * hidden2 + self.output_bias.expr()

        # return a list of outputs
        return output

    def train(self, train_file, epochs):
        # matplotlib config
        loss_values = []
        plt.ion()
        ax = plt.gca()
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 3])
        plt.title("Loss over time")
        plt.xlabel("Minibatch")
        plt.ylabel("Loss")
        createDictionary()

        for i in range(epochs):
            print('started epoch', (i+1))
            losses = []

            # train_file = "./data/train.data"
            train_data = open(train_file, 'r').read().strip().split('\n')

            # shuffle the training data.
            random.shuffle(train_data)

            step = 0
            for line in train_data:
                fields = line.strip().split(' ')
                features, label = fields[:-1], fields[-1]
                gold_label = action2id(label)
                result = self.forward(features, True)
		
                # getting loss with respect to negative log softmax function and the gold label.
                loss = dynet.pickneglogsoftmax(result, gold_label)

                # appending to the minibatch losses
                losses.append(loss)
                step += 1


                if len(losses) >= self.minibatch_size:
                    # now we have enough loss values to get loss for minibatch
                    minibatch_loss = dynet.esum(losses) / len(losses)

                    # calling dynetnet to run forward computation for all minibatch items
                    minibatch_loss.forward()

                    # getting float value of the loss for current minibatch
                    minibatch_loss_value = minibatch_loss.value()

                    # printing info and plotting
                    loss_values.append(minibatch_loss_value)
                    if len(loss_values)%10==0:
                        ax.set_xlim([0, len(loss_values)+10])
                        ax.plot(loss_values)
                        plt.draw()
                        plt.pause(0.0001)
                        progress = round(100 * float(step) / len(train_data), 2)
                        print('current minibatch loss', minibatch_loss_value, 'progress:', progress, '%')

                    # calling dynetnet to run backpropagation
                    minibatch_loss.backward()

                    # calling dynetnet to change parameter values with respect to current backpropagation
                    self.updater.update()

                    # empty the loss vector
                    losses = []

                    # refresh the memory of dynetnet
                    dynet.renew_cg()

            # there are still some minibatch items in the memory but they are smaller than the minibatch size
            # so we ask dynet to forget them
            dynet.renew_cg()

    def score(self, str_features):
        '''
        :param str_features: String features
        20 first: words, next 20: pos, next 12: dependency labels.
        DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
        :return: list of scores
        '''
        # change this part of the code.

        scores = self.forward(str_features, False).value()
        return scores

    def load(self, filename):
        self.nnmodel.populate(filename)

    def save(self, filename):
        self.nnmodel.save(filename)

if __name__=='__main__':
    # constructing network
    network = DepModel()

    # training
    network.train("./data/train.data", 12)

    # saving network
    network.save("./data/depModel")

    network = DepModel()  # creating default network

    # initially it was m = DepModel(), loading the trained model now
    network.load("./data/depModel")

    input_p = os.path.abspath(sys.argv[1])
    output_p = os.path.abspath(sys.argv[2])
    Decoder(network.score, network.actions).parse(input_p, output_p)
