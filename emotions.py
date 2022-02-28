import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import *

__author__ = 'kqureshi'


model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


def get_sentiment(text: str) -> float:
    """ 
    Usage: get_sentiment(text='Today is a great day')
    :param text:
    :param model:
    :param tokenizer:
    :return:
    """
    inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    output = model(input_ids, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
    logits = np.array(output.logits.tolist()[0])
    prob = np.exp(logits) / np.sum(np.exp(logits))
    return np.sum([(x + 1) * prob[x] for x in range(len(prob))])


