"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
text = open('input.txt', 'r').read() # should be simple plain text file
unique_chars = list(set(text))
text_length, vocab_size = len(text), len(unique_chars)
print('data has %d characters, %d unique.' % (text_length, vocab_size))
char_to_index = { char: i for i, char in enumerate(unique_chars) }
index_to_char = { i: char for i, char in enumerate(unique_chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
sequence_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
weights_input_to_hidden  = np.random.randn(hidden_size, vocab_size)  * 0.01 # input  -> hidden
weights_hidden_to_hidden = np.random.randn(hidden_size, hidden_size) * 0.01 # hidden -> hidden
weights_hidden_to_output = np.random.randn(vocab_size, hidden_size)  * 0.01 # hidden -> output
bias_hidden = np.zeros((hidden_size, 1)) # hidden bias
bias_output = np.zeros((vocab_size, 1)) # output bias

def compute_loss_and_gradients(input_indices, target_indices, previous_hidden_state):
  """
  input_indices, target_indices are both lists of integers.
  previous_hidden_state is an (H, 1) array of the initial hidden state.
  Returns the loss, gradients on model parameters, and last hidden state.
  """
  inputs_one_hot, hidden_states, output_logits, output_probs = {}, {}, {}, {}
  hidden_states[-1] = np.copy(previous_hidden_state)
  loss = 0
  # forward pass
  for t in range(len(input_indices)):
    inputs_one_hot[t] = np.zeros((vocab_size, 1)) # encode in 1-of-k representation
    inputs_one_hot[t][input_indices[t]] = 1
    hidden_states[t] = np.tanh(
        np.dot(weights_input_to_hidden,  inputs_one_hot[t]) +
        np.dot(weights_hidden_to_hidden, hidden_states[t-1]) +
        bias_hidden
    ) # hidden state
    output_logits[t] = np.dot(weights_hidden_to_output, hidden_states[t]) + bias_output # unnormalized log probabilities for next chars
    output_probs[t]  = np.exp(output_logits[t]) / np.sum(np.exp(output_logits[t])) # probabilities for next chars
    loss += -np.log(output_probs[t][target_indices[t], 0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  grad_weights_input_to_hidden  = np.zeros_like(weights_input_to_hidden)
  grad_weights_hidden_to_hidden = np.zeros_like(weights_hidden_to_hidden)
  grad_weights_hidden_to_output = np.zeros_like(weights_hidden_to_output)
  grad_bias_hidden = np.zeros_like(bias_hidden)
  grad_bias_output = np.zeros_like(bias_output)
  grad_hidden_next = np.zeros_like(hidden_states[0])
  for t in reversed(range(len(input_indices))):
    grad_output = np.copy(output_probs[t])
    grad_output[target_indices[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    grad_weights_hidden_to_output += np.dot(grad_output, hidden_states[t].T)
    grad_bias_output += grad_output
    grad_hidden = np.dot(weights_hidden_to_output.T, grad_output) + grad_hidden_next # backprop into h
    grad_hidden_raw = (1 - hidden_states[t] * hidden_states[t]) * grad_hidden # backprop through tanh nonlinearity
    grad_bias_hidden += grad_hidden_raw
    grad_weights_input_to_hidden  += np.dot(grad_hidden_raw, inputs_one_hot[t].T)
    grad_weights_hidden_to_hidden += np.dot(grad_hidden_raw, hidden_states[t-1].T)
    grad_hidden_next = np.dot(weights_hidden_to_hidden.T, grad_hidden_raw)
  for grad in [grad_weights_input_to_hidden, grad_weights_hidden_to_hidden,
               grad_weights_hidden_to_output, grad_bias_hidden, grad_bias_output]:
    np.clip(grad, -5, 5, out=grad) # clip to mitigate exploding gradients
  return (loss,
          grad_weights_input_to_hidden, grad_weights_hidden_to_hidden, grad_weights_hidden_to_output,
          grad_bias_hidden, grad_bias_output,
          hidden_states[len(input_indices) - 1])

def sample(hidden_state, seed_index, num_chars_to_sample):
  """
  Sample a sequence of integers from the model.
  hidden_state is the memory state, seed_index is the seed letter for the first time step.
  """
  input_one_hot = np.zeros((vocab_size, 1))
  input_one_hot[seed_index] = 1
  sampled_indices = []
  for t in range(num_chars_to_sample):
    hidden_state = np.tanh(
        np.dot(weights_input_to_hidden,  input_one_hot) +
        np.dot(weights_hidden_to_hidden, hidden_state) +
        bias_hidden
    )
    logits = np.dot(weights_hidden_to_output, hidden_state) + bias_output
    probs = np.exp(logits) / np.sum(np.exp(logits))
    next_char_index = np.random.choice(range(vocab_size), p=probs.ravel())
    input_one_hot = np.zeros((vocab_size, 1))
    input_one_hot[next_char_index] = 1
    sampled_indices.append(next_char_index)
  return sampled_indices

iteration, data_pointer = 0, 0
mem_weights_input_to_hidden  = np.zeros_like(weights_input_to_hidden)
mem_weights_hidden_to_hidden = np.zeros_like(weights_hidden_to_hidden)
mem_weights_hidden_to_output = np.zeros_like(weights_hidden_to_output)
mem_bias_hidden = np.zeros_like(bias_hidden)
mem_bias_output = np.zeros_like(bias_output) # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * sequence_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps sequence_length long)
  if data_pointer + sequence_length + 1 >= len(text) or iteration == 0:
    previous_hidden_state = np.zeros((hidden_size, 1)) # reset RNN memory
    data_pointer = 0 # go from start of data
  input_indices  = [char_to_index[char] for char in text[data_pointer    : data_pointer + sequence_length    ]]
  target_indices = [char_to_index[char] for char in text[data_pointer + 1: data_pointer + sequence_length + 1]]

  # sample from the model now and then
  if iteration % 100 == 0:
    sampled_indices = sample(previous_hidden_state, input_indices[0], 200)
    sampled_text = ''.join(index_to_char[i] for i in sampled_indices)
    print('----\n %s \n----' % (sampled_text,))

  # forward sequence_length characters through the net and fetch gradient
  (loss,
   grad_weights_input_to_hidden, grad_weights_hidden_to_hidden, grad_weights_hidden_to_output,
   grad_bias_hidden, grad_bias_output,
   previous_hidden_state) = compute_loss_and_gradients(input_indices, target_indices, previous_hidden_state)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if iteration % 100 == 0: print('iter %d, loss: %f' % (iteration, smooth_loss)) # print progress

  # perform parameter update with Adagrad
  for param, grad, mem in zip(
      [weights_input_to_hidden, weights_hidden_to_hidden, weights_hidden_to_output, bias_hidden, bias_output],
      [grad_weights_input_to_hidden, grad_weights_hidden_to_hidden, grad_weights_hidden_to_output, grad_bias_hidden, grad_bias_output],
      [mem_weights_input_to_hidden,  mem_weights_hidden_to_hidden,  mem_weights_hidden_to_output,  mem_bias_hidden,  mem_bias_output]):
    mem += grad * grad
    param += -learning_rate * grad / np.sqrt(mem + 1e-8) # adagrad update

  data_pointer += sequence_length # move data pointer
  iteration    += 1 # iteration counter
