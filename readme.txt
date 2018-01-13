Part1
python src/depModel.py trees/dev.conll trees/dev_part1.conll
python src/eval.py trees/dev.conll trees/dev_part1.conll
('Unlabeled attachment score', 83.74)
('Labeled attachment score', 80.59)

python src/depModel.py trees/test.conll trees/test_part1.conll

test_part1 has been copied in the output directory as mentioned.

------------------------------------------------------------------------------------------------------

Part2
python src/depModel.py trees/dev.conll trees/dev_part2.conll
python src/eval.py trees/dev.conll trees/dev_part2.conll
('Unlabeled attachment score', 84.07)
('Labeled attachment score', 80.91)

python src/depModel.py trees/test.conll trees/test_part2.conll

------------------------------------------------------------------------------------------------------

Part3

I tried a couple of different variations in the neural network parameters as follows:
hidden layer 1- 600, hidden layer 2- 600
minibatch_size- 2000
training_epochs- 12
MomentumSGDTrainer
('Unlabeled attachment score', 72.79)
('Labeled attachment score', 66.18)



Since accuracy achieved with MomentumSGDTrainer was quite low in comparison, I switched back to AdamTrainer
hidden layer 1- 600, hidden layer 2- 600
minibatch_size- 2000
training_epochs- 12
default biases
AdamTrainer
python src/eval.py trees/dev.conll trees/dev_partTemp2.conll
('Unlabeled attachment score', 83.92)
('Labeled attachment score', 80.62)



hidden layer 1- 600, hidden layer 2- 600
minibatch_size- 1000
training_epochs- 12
biases- 0.2 for hidden layers, and 0 for output layer
AdamTrainer
python src/eval.py trees/dev.conll trees/dev_partTemp3.conll
('Unlabeled attachment score', 84.64)
('Labeled attachment score', 81.38)



hidden layer 1- 600, hidden layer 2- 600
minibatch_size- 1000
training_epochs- 12
biases- 0.2 for hidden layers, and 0 for output layer
AdamTrainer
leaky relu as the transfer function
inverted-dropout, with probability 0.2
glove 100d word embeddings are used for the words that were available in glove


python src/eval.py trees/dev.conll trees/dev_partTemp4.conll
('Unlabeled attachment score', 84.55)
('Labeled attachment score', 81.14)