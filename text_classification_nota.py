# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import textattack
import argparse
import os
import math
import collections

from src.dataset import load_data
from src.utils import bool_flag

from textattack.shared.utils import logger
from textattack.attack import Attack
from textattack.attack_args import AttackArgs
from textattack.attack_results import MaximizedAttackResult, SuccessfulAttackResult, FailedAttackResult
from textattack.attacker import Attacker
from textattack.model_args import HUGGINGFACE_MODELS
from textattack.models.helpers import LSTMForClassification, WordCNNForClassification
from textattack.models.wrappers import ModelWrapper
from textattack.training_args import CommandLineTrainingArgs, TrainingArgs
from a2t import A2TYoo2021
class Trainer_NOTA(textattack.Trainer):
    def _generate_adversarial_examples(self, epoch):
        """Generate adversarial examples using attacker."""
        assert (
            self.attack is not None
        ), "`attack` is `None` but attempting to generate adversarial examples."
        base_file_name = f"attack-train-{epoch}"
        log_file_name = os.path.join(self.training_args.output_dir, base_file_name)
        logger.info("Attacking model to generate new adversarial training set...")

        if isinstance(self.training_args.num_train_adv_examples, float):
            num_train_adv_examples = math.ceil(
                len(self.train_dataset) * self.training_args.num_train_adv_examples
            )
        else:
            num_train_adv_examples = self.training_args.num_train_adv_examples

        # Use Different AttackArgs based on num_train_adv_examples value.
        # If num_train_adv_examples >= 0 , num_train_adv_examples is
        # set as number of successful examples.
        # If num_train_adv_examples == -1 , num_examples is set to -1 to
        # generate example for all of training data.
        if num_train_adv_examples >= 0:
            attack_args = AttackArgs(
                num_successful_examples=num_train_adv_examples,
                num_examples_offset=0,
                query_budget=self.training_args.query_budget_train,
                shuffle=True,
                parallel=self.training_args.parallel,
                num_workers_per_device=self.training_args.attack_num_workers_per_device,
                disable_stdout=True,
                silent=True,
                log_to_txt=log_file_name + ".txt",
                log_to_csv=log_file_name + ".csv",
            )
        elif num_train_adv_examples == -1:
            # set num_examples when num_train_adv_examples = -1
            attack_args = AttackArgs(
                num_examples=num_train_adv_examples,
                num_examples_offset=0,
                query_budget=self.training_args.query_budget_train,
                shuffle=True,
                parallel=self.training_args.parallel,
                num_workers_per_device=self.training_args.attack_num_workers_per_device,
                disable_stdout=True,
                silent=True,
                log_to_txt=log_file_name + ".txt",
                log_to_csv=log_file_name + ".csv",
            )
        else:
            assert False, "num_train_adv_examples is negative and not equal to -1."

        attacker = Attacker(self.attack, self.train_dataset, attack_args=attack_args)
        results = attacker.attack_dataset()

        attack_types = collections.Counter(r.__class__.__name__ for r in results)
        total_attacks = (
            attack_types["SuccessfulAttackResult"] + attack_types["FailedAttackResult"]
        )
        success_rate = attack_types["SuccessfulAttackResult"] / total_attacks * 100
        logger.info(f"Total number of attack results: {len(results)}")
        logger.info(
            f"Attack success rate: {success_rate:.2f}% [{attack_types['SuccessfulAttackResult']} / {total_attacks}]"
        )
        # TODO: This will produce a bug if we need to manipulate ground truth output.

        # To Fix Issue #498 , We need to add the Non Output columns in one tuple to represent input columns
        # Since adversarial_example won't be an input to the model , we will have to remove it from the input
        # dictionary in collate_fn
        
        nota_label = torch.max(self.train_dataset.labels) + 1
        
        adversarial_examples = [
            (
                tuple(r.perturbed_result.attacked_text._text_input.values())
                + ("adversarial_example",),
                nota_label,#r.perturbed_result.ground_truth_output,
            )
            for r in results
            if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult,FailedAttackResult))
        ]

        # Name for column indicating if an example is adversarial is set as "_example_type".
        adversarial_dataset = textattack.datasets.Dataset(
            adversarial_examples,
            input_columns=self.train_dataset.input_columns + ("_example_type",),
            label_map=self.train_dataset.label_map,
            label_names=self.train_dataset.label_names,
            output_scale_factor=self.train_dataset.output_scale_factor,
            shuffle=False,
        )
        return adversarial_dataset

def main(args):
    
    dataset, num_labels = load_data(args)
    num_labels += 1    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    if args.model == 'gpt2':
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    if args.dataset == "mnli":
        # only evaluate on matched validation set
        testset_key = "validation_matched"
        preprocess_function = lambda examples: tokenizer(
            examples["premise"], examples["hypothesis"], max_length=256, truncation=True)
    else:
        text_key = 'text' if (args.dataset in ["ag_news", "imdb", "yelp"]) else 'sentence'
        testset_key = 'test' if (args.dataset in ["ag_news", "imdb", "yelp"]) else 'validation'
        preprocess_function = lambda examples: tokenizer(examples[text_key], max_length=256, truncation=True)
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    training_args = textattack.TrainingArgs(
        num_epochs=args.epochs,
        num_clean_epochs=1,
        attack_epoch_interval=1,
        parallel=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        #num_warmup_steps=args.num_warmup_steps,
        learning_rate=args.lr,
        num_train_adv_examples=1,#0.2,
        attack_num_workers_per_device=6,
        query_budget_train=200,
        checkpoint_interval_epochs=1,
        output_dir=args.checkpoint_folder,
        log_to_wandb=False,
        weight_decay=args.weight_decay,
        #wandb_project="nlp-robustness",
        load_best_model_at_end=True,
        logging_interval_step=10,
        random_seed=42,
    )

    if not args.finetune:
        # freeze parameters of transformer
        transformer = list(model.children())[0]
        for param in transformer.parameters():
            param.requires_grad = False

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    
    trainer = Trainer_NOTA(
        model_wrapper,
        "classification",
        attack= A2TYoo2021.build(model_wrapper, mlm=False),
        train_dataset=textattack.datasets.huggingface_dataset.HuggingFaceDataset(encoded_dataset["train"]),
        eval_dataset=textattack.datasets.huggingface_dataset.HuggingFaceDataset(encoded_dataset[testset_key]),
        training_args=training_args,
    )

    trainer.train()
    trainer.evaluate()
    suffix = ''
    if args.finetune:
        suffix += '_finetune'
    torch.save(model.state_dict(),
               os.path.join(args.result_folder, "%s_%s%s_nota.pth" % (args.model.replace('/', '-'), args.dataset, suffix)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text classification model training.")

    # Bookkeeping
    parser.add_argument("--checkpoint_folder", default="checkpoint/", type=str,
        help="folder in which to store temporary model checkpoints")
    parser.add_argument("--result_folder", default="result/", type=str,
        help="folder in which to store trained models")
    parser.add_argument("--tqdm", default=True, type=bool_flag,
        help="Use tqdm in output")

    # Data 
    parser.add_argument("--data_folder", required=True, type=str,
        help="folder in which to store data")
    parser.add_argument("--dataset", default="dbpedia14", type=str,
        choices=["dbpedia14", "ag_news", "imdb", "yelp", "mnli"],
        help="classification dataset to use")

    # Model
    parser.add_argument("--model", default="gpt2", type=str,
        help="type of model")

    # Optimization
    parser.add_argument("--batch_size", default=16, type=int,
        help="batch size for training and evaluation")
    parser.add_argument("--epochs", default=4, type=int,
        help="number of epochs to train for")
    parser.add_argument("--lr", default=2e-5, type=float,
        help="learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float,
        help="weight decay")
    parser.add_argument("--finetune", default=False, type=bool_flag,
        help="finetune the transformer; if False, only train linear layer")

    args = parser.parse_args()

    if args.result_folder == 'none':
        args.result_folder = args.checkpoint_folder

    main(args)