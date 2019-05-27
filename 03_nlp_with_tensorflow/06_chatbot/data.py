import tensorflow as tf
import pandas as pd
import numpy as np
import re


import tqdm
from configs import DEFINES
from sklearn.model_selection import train_test_split
#from konlpy.tag import Okt
from konlpy.tag import Kkma



FILTERS = "([~.,!?\"':;)(])"
PAD = "<PADDING>"
STD = "<START>"
END = "<END>"
UNK = "<UNKNOWN>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)


def load_data():
  data_df = pd.read_csv(DEFINES.data_path, header=0)
  question, answer = list(data_df['Q']), list(data_df['A'])
  train_input, eval_input, train_label, eval_label = train_test_split(question, answer, test_size=0.33, random_state=42)
  return train_input, train_label, eval_input, eval_label

def prepro_like_morphlized(data):
  morph_analyzdr = Kkma()
  result_data = list()
  for seq in tqdm(data):
    morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ','')))
    result_data.append(morphlized_seq)
  
  return result_data

def enc_processing(value, dictionary):
  sequences_input_index = []
  sequences_length = []
  
  if DEFINES.tokenize_as_morph:
    value = prepro_like_morphlized(value)
  
  for sequence in value:
    sequence = re.sub(CHANGE_FILTER, "", sequence)
    sequence_index = []
    for word in sequence.split():
      if dictionary.get(word) is not None:
        sequence_index.extend([dictionary[word]])
      else:
        sequence_index.extend([dictionary[UNK]])
    
    if len(sequence_index) > DEFINES.max_sequence_length:
      sequence_index = sequence_index[:DEFINES.max_sequence_length]

    sequences_length.append(len(sequence_index))
    sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
    sequences_input_index.append(sequence_index)

  return np.asarray(sequences_input_index), sequences_length



def dec_input_processing(value, dic):
  sequences_output_index = []
  sequences_length = []

  if DEFINES.tokenize_as_morph:
    value = prepro_like_morphlized(value)

  for sequence in value:
    sequence = re.sub(CHANGE_FILTER, "", sequence)
    sequence_index = []
    sequence_index = [dictionary[STD]] + [dictionary[word] for word in sequence.split()]

    if len(sequence_index) > DEFINES.max_sequence_length:
      sequence_index = sequence_index[:DEFINES.max_sequence_length]
    
    sequence_length.append(len(sequence_index))
    sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
    sequences_output_index.append(sequence_index)
  
  return np.asarray(sequences_output_index), sequences_length


def dec_target_processing(value, dectionary):
  sequences_target_index = []

  if DEFINES.tokenize_as_morph:
    value = prepro_like_morphlized(value)

  for sequence in value:
    sequence = re.sub(CHANGE_FILTER, "", sequence)
    sequence_index = [dictionary[word] for word in sequence.split()]
    if len(sequence_index) >= DEFINES.max_sequence_length:
      sequence_index = sequence_index[:DEFINES.max_sequence_length-1] + [dictionary[END]]
    else:
      sequence_index += [dictionary[END]]
    
    sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
    sequence_target_index.append(sequence_index)

  return np.asarray(sequences_target_index)


def pred2string(value, dictionary):
  sentence_string = []
  for v in value:
    sentence_string = [dictionary[index] for index in v['indexs']]

  print(sentence_string)
  answer = ""

  for word in sentence_string:
    if word not in PAD and word not in END:
      answer += word
      answer += " "

  print(answer)
  return answer



def data_tokenizer(data):
  words = []
  for sentence in data:
    sentence = re.sub(CHANGE_FILTER, "", sentence)
    for word in sentence.split():
      words.append(wored)
  return [word for word in words if word]


def load_vocabulary():
  vocabulary_list = []
  if (not (os.path.exists(DEFINES.vocabulary_path))):
    if (os.path.exists(DEFINES.data_path)):
      data_df = pd.read_csv(DEFINES.data_path, encoding='utf-8')
      question, answer = list(data_df['Q']), list(data_df['A'])
      if DEFINES.tokenize_as_morph:
        question = prepro_like_morphlized(question)
        answer = prepro_like_morphlized(answer)
      
      data = []
      data.extend(question)
      data.extend(answer)
      words = data_tokenizer(data)
      words = list(set(words))
      words[:0] = MARKER

    with open(DEFINES.vocabulary_path, 'w') as vocabulary_file:
      for word in words:
        vocabulary_file.write(word + '\n')
  with open(DEFINES.vocabulary_path, 'r', encoding='utf-8') as vocabulary_file:
    for line in vocabulary_file:
      vocabulary_list.append(line.strip())
  
  word2idx, idx2word = make_vocabulary(vocabulary_list)

def make_vocabulary(vocabulary_list):
  word2idx = {word: idx for idx, word in enumerate(vocabulary_list)}
  idx2word = {idx: word for idx, word in enumerate(vocabulary_list)}
  return word2idx, idx2word


def train_input_fn(train_input_enc, train_output_dec, train_target_dec, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices((train_input_enc, train_output_dec, train_target_dec))
  dataset = dataset.shuffle(buffer_size=len(train_input_enc))
  assert batch_size is not None, "train batchSize must not be None"
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(rearrange)
  dataset = dataset.repeat()
  iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()


def eval_input_fn(eval_input_enc, eval_output_dec, eval_target_dec, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices((eval_input_enc, eval_input_dec, eval_target_dec))
  dataset = dataset.shuffle(buffer_size = len(eval_input_enc))
  assert batch_size is not None, "eval batchSize must not be None"
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(rearrange)
  dataset = dataset.repeat(1)
  iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()

def rearrange(input, output, target):
  features = {"input":input, "output":output}
  return features, target

