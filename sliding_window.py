import argparse
import json
from collections import Counter
from multiprocessing import Pool

import tqdmlogger as logger
import numpy as np
import nltk
from tqdm import tqdm

rm_stop = False
PUNCTS = ['.', '?', ',', '!', '"', '\'', '$', '%', '^', '&']

token_mappers = []
stopwords = open('./stopwords.txt', 'r').read().split('\n')
stopwords = set(map(lambda x: x.strip().rstrip(), stopwords))
token_mappers.append(lambda x: x if x.lower() not in stopwords else None)
token_mappers.append(lambda x: x if x not in PUNCTS else None)

def tokenize(text):
    text = text.replace('\n', ' ')
    mapped = nltk.word_tokenize(text)
    if rm_stop:
        for mapper in token_mappers:
            mapped = list(filter(lambda x: x is not None, map(mapper, mapped)))
    return mapped

def compute_counts(examples):
    counts = Counter()
    for e in tqdm(examples, desc="compute num of tokens"):
        counts.update(tokenize(e.contexts[0]))
    return counts


def compute_inverse_counts(stories):
    counts = compute_counts(stories)
    icounts = {}
    for token, token_count in counts.items():
        icounts[token] = np.log(1.0 + 1.0 / token_count)
    return icounts


def baseline_distance(passage, question, answer):
    if not isinstance(question, set):
        question = set(question)
    if not isinstance(answer, set):
        answer = set(answer)
    s_question = question.intersection(passage)
    s_answer = answer.intersection(passage).difference(question)
    if len(s_question) == 0 or len(s_answer) == 0:
        return 1.0
    last_q, last_a = np.inf, np.inf
    closest = np.inf
    for i, token in enumerate(passage):
        if token in s_question:
            last_q = i
        if token in s_answer:
            last_a = i
        if abs(last_q - last_a) < closest:
            # print(last_q, last_a)
            closest = np.abs(last_q - last_a) / (len(passage) - 1)
    assert closest > 0 and closest <= 1
    return closest


class SlidingWindow(object):
    def __init__(self):
        pass

    def fit(self, examples, window_size=None):
        self._icounts = compute_inverse_counts(examples)
        self._window_size = window_size

    def predict_target(self, tokens, target, verbose=True):
        if not isinstance(target, set):
            target = set(target)
        target_size = len(target)
        max_overlap_score = 0.0
        tokens_at_max = []
        for i in range(len(tokens)):
            overlap_score = 0.0
            try:
                window_size = self._window_size or target_size
                for j in range(window_size):
                    if tokens[i + j] in target:
                        overlap_score += self._icounts[tokens[i + j]]
            except IndexError:
                pass
            if overlap_score > max_overlap_score:
                tokens_at_max = tokens[i:i + window_size]
                max_overlap_score = overlap_score
        if verbose:
            print('[score=%.2f for target=%s] passage: %s ' %
                  (max_overlap_score, target, tokens_at_max))
        return max_overlap_score

    def predict(self, passage, question, answers, with_distance=False, verbose=False):
        scores = []
        if verbose:
            print('Question: %s' % question)
        for answer in answers:
            dist = baseline_distance(passage, question, answer)if with_distance else 0
            scores.append(self.predict_target(passage, set(question + answer), verbose) - dist)
        return scores

def get_predicts_score_helper(example):
    context_tokens = tokenize(example.contexts[0]) 
    question_tokens = tokenize(example.question) 
    options_tokens = map(lambda x: tokenize(x), example.endings)
    score = sliding_window.predict(
        context_tokens, question_tokens, options_tokens,
        with_distance=distance, verbose=False,
    )
    return example.example_id, score


def get_predicts_score(examples, sliding_window, with_distance=False):
    def init(sw, d):
        global sliding_window
        sliding_window = sw
        global distance
        distance = d
    with Pool(8, initializer=init,  initargs=(sliding_window, with_distance)) as p:
        predicted = dict(
            tqdm(
                p.imap(get_predicts_score_helper, examples, chunksize=32),
                total=len(examples),
                desc="sliding window",
            )
        )
    return predicted
