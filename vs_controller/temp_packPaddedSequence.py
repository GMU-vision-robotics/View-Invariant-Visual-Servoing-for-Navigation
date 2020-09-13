import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

seqs = ['gigantic_string','tiny_str','medium_str']

# make <pad> idx 0
vocab = ['<pad>'] + sorted(set(''.join(seqs)))

# make model
embed = nn.Embedding(len(vocab), 10).cuda()
lstm = nn.LSTM(10, 5, batch_first=True).cuda()
linear = nn.Linear(5, 3).cuda()


vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]

# get the length of each seq in your batch
seq_lengths = torch.LongTensor([len(seq) for seq in vectorized_seqs]).cuda()

# dump padding everywhere, and place seqs on the left.
# NOTE: you only need a tensor as big as your longest sequence
seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long().cuda()
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
	seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

# SORT YOUR TENSORS BY LENGTH!
## seq_lengths is the length of the sorted subsequences.
## perm_idx is the index of the sorted sequences.
seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
seq_tensor = seq_tensor[perm_idx]

# utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
# Otherwise, give (L,B,D) tensors
#seq_tensor = seq_tensor.transpose(0, 1) # (B,L,D) -> (L,B,D)

print('seq_tensor.shape = {}'.format(seq_tensor.shape))

# embed your sequences
seq_tensor = embed(seq_tensor)
print('seq_tensor.shape = {}'.format(seq_tensor.shape))

# pack them up nicely
packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)

# throw them through your LSTM (remember to give batch_first=True here if you packed with it)
packed_output, (ht, ct) = lstm(packed_input)
packed_output, seq_lengths = pad_packed_sequence(packed_output, batch_first=True)
output = linear(packed_output)
# unpack your output if required
#output, _ = pad_packed_sequence(packed_output, batch_first=True)
print (output)
assert 1==2

# Or if you just want the final hidden state?
print (ht[-1])

# REMEMBER: Your outputs are sorted. If you want the original ordering
# back (to compare to some gt labels) unsort them
_, unperm_idx = perm_idx.sort(0)
output = output[unperm_idx]
print (output)