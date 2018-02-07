# -*- coding: utf-8 -*-

# Also you can clear the default graph from memory
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Saving data pipeline structures (vocabulary, )
word_list = ['to', 'be', 'or', 'not', 'to', 'be']
vocab_list = list(set(word_list))
vocab2ix_dict = dict(zip(vocab_list, range(len(vocab_list))))
ix2vocab_dict = {val:key for key,val in vocab2ix_dict.items()}

# Save vocabulary
import json
with open('vocab2ix_dict.json', 'w') as file_conn:
    json.dump(vocab2ix_dict, file_conn)

# Load vocabulary
with open('vocab2ix_dict.json', 'r') as file_conn:
    vocab2ix_dict = json.load(file_conn)

print(vocab2ix_dict)
    
print('finished')    