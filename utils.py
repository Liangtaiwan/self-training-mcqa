# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """


import csv
import glob
import json
import logging
import os
from functools import partial
from typing import List

import tqdmlogger as logger
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from fuzzywuzzy import process
import numpy as np

from sliding_window import SlidingWindow, get_predicts_score

tqdm = partial(tqdm, ncols=80)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.example_id}",
            f"article: {self.contexts}",
            f"question: {self.question}",
            f"option_0: {self.endings[0]}",
            f"option_1: {self.endings[1]}",
            f"option_2: {self.endings[2]}",
            f"option_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""
    
    def get_examples(self, data_dir, data_type):
        if data_type=='train':
            return self.get_train_examples(data_dir)
        elif data_type=='dev':
            return self.get_dev_examples(data_dir)
        elif data_type=='test':
            return self.get_test_examples(data_dir)
        else:
            raise ValueError("data_type should be in [train, dev, test]")

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def load_predictions(self, data_dir, data_type):
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        json_file = os.path.join(data_dir, "train.json")
        return self._read_json_examples(json_file)

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        json_file = os.path.join(data_dir, "dev.json")
        return self._read_json_examples(json_file)

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        json_file = os.path.join(data_dir, "test.json")
        return self._read_json_examples(json_file)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json_examples(self, input_file):
        """Read a SQuAD format json file into a list of RACEExample."""
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"]
                    options = qa["options"]
                    answer = qa["answers"][0]["answer"]
                    label = str(ord(answer) - ord('A'))
                    example = InputExample(
                        example_id=qas_id,
                        question=question,
                        contexts=[context, context, context, context],
                        endings=options,
                        label=label,
                    )
                    examples.append(example)
        return examples

    def load_predictions(self, data_dir, data_type):
        with open(os.path.join(data_dir, f"predictions_{data_type}.json")) as fin:
            predictions = json.load(fin)
        return predictions

class MCTestProcessor(DataProcessor):
    """Processor for the RACE data set."""
    
    def __init__(self):
        self._num_story = 500

    @property
    def num_story(self):
        return self._num_story

    @num_story.setter
    def num_story(self, new_num_story):
        if new_num_story in [160, 500]:
            self._num_story = new_num_story
        else:
            raise ValueError("num_story can only be 160 or 500")

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        json_file = os.path.join(data_dir, f"mc{self._num_story}.train.json")
        return self._read_json_examples(json_file)

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        json_file = os.path.join(data_dir, f"mc{self._num_story}.dev.json")
        return self._read_json_examples(json_file)

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        json_file = os.path.join(data_dir, f"mc{self._num_story}.test.json")
        return self._read_json_examples(json_file)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json_examples(self, input_file):
        """Read a SQuAD format json file into a list of MCTestEExample."""
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"]
                    options = qa["options"]
                    answer = qa["answers"][0]["answer"]
                    label = str(ord(answer) - ord('A'))
                    example = InputExample(
                        example_id=qas_id,
                        question=question,
                        contexts=[context, context, context, context],
                        endings=options,
                        label=label,
                    )
                    examples.append(example)
        return examples
    
    def load_predictions(self, data_dir, data_type):
        with open(os.path.join(data_dir, f"predictions_mc{self.num_story}.{data_type}.json")) as fin:
            predictions = json.load(fin)
        return predictions


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer.encode_plus(
                text_a, text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))
        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
                logger.info("label: {}".format(label))

        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label,))

    return features


processors = {"race": RaceProcessor, "mctest": MCTestProcessor}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race", 4, "mctest", 4}


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def accuracy(preds, labels, examples, task):
    if task=="race":
        type1, type2 = "middle", "high"
    elif task=="mctest":
        type1, type2 = "one", "multiple"
    else:
        NotImplementedError
    num_type1 = 0
    num_type2 = 0
    score_type1 = 0
    score_type2 = 0
    for pred, label, e in zip(preds, labels, examples):
        if type1 in e.example_id:
            num_type1 += 1
            score_type1 += (pred == label)
        elif type2 in e.example_id:
            num_type2 += 1
            score_type2 += (pred == label)
        else:
            KeyError("Problem type only type2 or type1")
    type1_acc = score_type1 / num_type1 if num_type1 != 0 else 0
    type2_acc = score_type2 / num_type2 if num_type2 != 0 else 0
    assert num_type1 + num_type2 == len(examples) 
    acc = (score_type2 + score_type1) / (num_type2 + num_type1)
    return {"eval_acc": acc, f"{type1}_acc": type1_acc, f"{type2}_acc": type2_acc}

class Extractor:
    "Base class for extract candidates"
    def extract_candidates(self, examples, predictions, n=1, threshold=None):
        raise NotImplementedError()


class FuzzyMatchExtractor(Extractor):
    def extract_candidates(self, examples, predictions, n=1, threshold=None, return_scores=False):
        candidates_dict = {}
        for e in tqdm(examples):
            candidates = process.extract(predictions[e.example_id], e.endings)
            if threshold is not None:
                if return_scores:
                    candidates_dict[e.example_id] = [(e.endings.index(c[0]), c[1]) for c in candidates if c[1]>=threshold][:n]
                else:
                    candidates_dict[e.example_id] = [e.endings.index(c[0]) for c in candidates if c[1]>=threshold][:n]
            else:
                if return_scores:
                    candidates_dict[e.example_id] = [(e.endings.index(c[0]), c[1]) for c in candidates][:n]
                else:
                    candidates_dict[e.example_id] = [e.endings.index(c[0]) for c in candidates][:n]
        logger.info("Average number of candidates:", np.mean([len(c) for c in candidates_dict.values()]))
        return candidates_dict


class SlidingWindowExtractor(Extractor):
    init_sw = True
    def extract_candidates(self, examples, n=1, threshold=None, with_distance=False, return_scores=False):
        candidates_dict = {}
        if self.init_sw:
            self.sw = SlidingWindow()
            self.sw.fit(examples)
            self.init_sw = False
        predictions = get_predicts_score(examples, self.sw, with_distance)
        for e in tqdm(examples):
            candidates = list(zip(e.endings, predictions[e.example_id]))
            candidates.sort(key=lambda x: x[1], reverse=True)
            if threshold is not None:
                if return_scores:
                    candidates_dict[e.example_id] = [(e.endings.index(c[0]), c[1]) for c in candidates if c[1]>=threshold/100][:n]
                else:
                    candidates_dict[e.example_id] = [e.endings.index(c[0]) for c in candidates if c[1]>=threshold/100][:n]
            else:
                if return_scores:
                    candidates_dict[e.example_id] = [(e.endings.index(c[0]), c[1]) for c in candidates][:n]
                else:
                    candidates_dict[e.example_id] = [e.endings.index(c[0]) for c in candidates][:n]
        logger.info("Average number of candidates:", np.mean([len(c) for c in candidates_dict.values()]))
        return candidates_dict


extractors = {"sw": SlidingWindowExtractor, "fz": FuzzyMatchExtractor}

