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
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""


import glob
import os
import random
import time
from functools import partial

try:
    from comet_writer import CometWriter
except:
    pass
import jsonargparse as argparse
from jsonargparse import ActionConfigFile
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import tqdmlogger as logger
from tqdm import tqdm, trange
import torch_optimizer as optim
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)


from utils import convert_examples_to_features, processors, extractors, accuracy 
from modeling import BertForMultipleChoice, RobertaForMultipleChoice


tqdm = partial(tqdm, ncols=80)
trange = partial(trange, ncols=80)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)), ()
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMultipleChoice, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer),
}


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, tb_writer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.optimizer.lower() == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    elif args.optimizer.lower() == 'radam':
        optimizer = optim.RAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        raise NotImplementedError()

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = {}".format(
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
        )
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_test_acc = 0.0
    best_steps = 0
    best_test_steps = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    rand_num = np.random.rand()
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2]
                if args.model_type in ["bert", "xlnet"]
                else None,  # XLM don't use segment_ids
                "labels": batch[3],
                "answer_mask": batch[4] if args.label_type == 'match' else None,
            }
            outputs = model(**inputs, global_step=global_step+1, rand_num=rand_num)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            with logger.updating():
                logger.seclog(['Loss', 'blue'], loss.item())
                logger.seclog(['Global Steps', 'blue'], global_step+1)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if args.optimizer.lower() == 'adamw':
                    scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                rand_num = np.random.rand()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    logger.seclog(
                        ['Average Loss', 'blue'],
                        str((tr_loss - logging_loss) / args.logging_steps),
                    )

                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, prefix=f'global_step: {global_step}')
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if results["eval_acc"] > best_dev_acc:
                            best_dev_acc = results["eval_acc"]
                            best_steps = global_step
                        if args.do_test:
                            results_test = evaluate(args, model, tokenizer, test=True)
                            for key, value in results_test.items():
                                tb_writer.add_scalar("test_{}".format(key), value, global_step)
                            logger.info(
                                "test acc: {}, loss: {}, global steps: {}".format(
                                str(results_test["eval_acc"]),
                                str(results_test["eval_loss"]),
                                str(global_step),
                                )
                            )
                    if args.optimizer.lower == 'adamw':
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)

                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)

                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.flush()

    return global_step, tr_loss / global_step, best_steps, best_dev_acc


def evaluate(args, model, tokenizer, prefix="", test=False):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, examples = load_and_cache_examples(args, eval_task, tokenizer, evaluate=not test, test=test)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size = {args.eval_batch_size}")
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                    if args.model_type in ["bert", "xlnet"]
                    else None,  # XLM don't use segment_ids
                    "labels": batch[3],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        # acc = simple_accuracy(preds, out_label_ids)
        # result = {"eval_acc": acc, "eval_loss": eval_loss}
        result = accuracy(preds, out_label_ids, examples, args.task_name)
        result.update(eval_loss=eval_loss)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(str(prefix) + " is test:" + str(test)))
            writer.write(f"model           ={args.model_name_or_path}\n")
            writer.write(
                "total batch size=%d\n"
                % (
                    args.per_gpu_train_batch_size
                    * args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
                )
            )
            writer.write(f"train num epochs={args.num_train_epochs}\n")
            writer.write(f"fp16            ={args.fp16}\n")
            writer.write(f"max seq length  ={args.max_seq_length}\n")
            for key in sorted(result.keys()):
                logger.info("  {} = {}".format(key, str(result[key])))
                writer.write(f"{key} = {result[key]}\n")
    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    extractor = extractors[args.matching_method]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    if args.do_train:
        model_name = list(filter(None, args.model_name_or_path.split("/"))).pop(),
    else:
        model_name = "bert-base-uncased"
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )

    if evaluate:
        examples = processor.get_dev_examples(args.data_dir)
    elif test:
        examples = processor.get_test_examples(args.data_dir)
    else:
        examples = processor.get_train_examples(args.data_dir)

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating features from dataset file at {args.data_dir}")
        label_list = processor.get_labels()

        logger.info(f"Training number: {len(examples)}")
        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token_segment_id=tokenizer.pad_token_type_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if args.label_type == 'match' and not evaluate and not test:
        if args.threshold is None:
            logger.warning("threshold is None")
        max_n_candidates = args.max_n_candidates
        examples = processor.get_train_examples(args.data_dir)
        if args.matching_method=="fz":
            predictions =  processor.load_predictions(args.data_dir, "train")
            candidates = extractor.extract_candidates(
                examples,
                predictions,
                n=max_n_candidates,
                threshold=args.threshold,
            )
        elif args.matching_method=="sw":
            candidates = extractor.extract_candidates(
                examples,
                n=max_n_candidates,
                threshold=args.threshold,
                with_distance=args.distance,
            )
        features = [f for f in features if len(candidates[f.example_id])!=0] 
        for f in features:
            label = candidates[f.example_id]
            n_label = len(label)
            answer_mask = [1 for _ in range(n_label)]
            for _ in range(max_n_candidates-n_label):
                label.append(0) # append no answer
                answer_mask.append(0)
            f.label = label
            f.answer_mask = answer_mask
        all_answer_mask = torch.tensor([f.answer_mask for f in features], dtype=torch.long)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    if args.label_type == 'match' and not evaluate and not test:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_answer_mask)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset, examples


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("-opt", "--optimizer", default='adamw', type=str, help="Optimizers.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # My exp argparse
    parser.add_argument("--label_type", type=str)
    parser.add_argument("--matching_method", type=str, default="fz", choices=["fz", "sw"])
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--max_n_candidates", type=int, default=4)
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--writer", type=str, default="tensorboard")
    parser.add_argument("--sync_tensorboard", type=bool, default=False)  # This is only availabel with jsonargparse.
    parser.add_argument("--distance", type=bool, default=False)  # This is only availabel with jsonargparse.
    
    # config file
    parser.add_argument('--cfg', action=ActionConfigFile)
    parser.add_argument('--data_cfg', action=ActionConfigFile)
    parser.add_argument('--loss_cfg', action=ActionConfigFile)
    args = parser.parse_args()
    
    if args.loss_type == "hard-em":
        dir_name = f"{args.task_name}_{args.loss_type}_{args.threshold}_{args.max_n_candidates}_{args.tau}"
    else:
        dir_name = f"{args.task_name}_{args.loss_type}_{args.threshold}_{args.max_n_candidates}"
    args.output_dir = os.path.join(
        args.output_dir,
        dir_name,
    )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    # )
    logger.warning(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
        )
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.update({"loss_type": args.loss_type, "tau": args.tau})
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    
    logger.info(f"Training/evaluation parameters")
    for key, value in vars(args).items():
        logger.seclog([key, 'light_blue'], value)

    if args.local_rank in [-1, 0]:
        if args.writer == "tensorboard":
            tb_writer = SummaryWriter(f"./runs/{args.loss_type}_{args.threshold}_{time.strftime('%Y%m%d%H%M%S')}")
        elif args.writer == "comet":
            tb_writer = CometWriter(
                project_name="unsupervised-mrqa",
                workspace="liangtaiwan",
                exp_name=f"{args.loss_type}_{args.threshold}",
                auto_param_logging=False,
                auto_metric_logging=False,
                auto_output_logging=False,
                sync_tensorboard=args.sync_tensorboard,
            )
            tb_writer.log_parameters(vars(args))
        else:
            raise NotImplementedError()
    best_steps = 0

    # Training
    if args.do_train:
        train_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss, best_steps, best_dev_acc = train(args, train_dataset, model, tokenizer, tb_writer)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

    # Saving last-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info(f"Saving model checkpoint to {args.output_dir}")
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
        logger.info("Evaluate the following checkpoints:", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
            for key, value in result.items():
                tb_writer.add_scalar("eval_{}".format(key), value)

    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        # if args.eval_all_checkpoints: # can not use this to do test!!
        #     checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        #     logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info(f"Evaluate the following checkpoints: {checkpoints}")
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix, test=True)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
            for key, value in result.items():
                tb_writer.add_scalar("test_{}".format(key), value)
            tb_writer.flush()
    if best_steps:
        logger.info(f"best steps of eval acc is the following checkpoints: {best_steps}")
        logger.info(f"best eval acc: {best_dev_acc}")
    return results


if __name__ == "__main__":
    main()
