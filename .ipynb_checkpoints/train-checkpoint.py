import os
os.environ['TRANSFORMERS_CACHE'] = '/data/tungtx2/tmp/transformers_hub'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
from pathlib import Path
import torch
from transformers import TrainingArguments, Trainer
from myutils import *
from dataset import *
from config import *
from model import *
import evaluate
import numpy as np
from seqeval.metrics import classification_report
import yaml
import shutil


if __name__ == '__main__':
    # load config
    config = get_config('training.yaml')
    os.makedirs(config.training.trainer_args.output_dir, exist_ok=True)
    shutil.copy('training.yaml', config.training.trainer_args.output_dir)

    # # check and correct if any json contain error
    # check_json_files(config.data.train_dir)
    # check_json_files(config.data.val_dir)

    # load labels
    train_labels = find_all_labels(config.data.train_dir, ls_exclude_dir=config.data.ls_exclude_dir)
    val_labels = find_all_labels(config.data.val_dir)
    print('TRAIN LABELS:')
    for label in train_labels:
        print('\t' + label)
    print('VAL LABELS:')
    for label in val_labels:
        print('\t' + label)
    assert val_labels.issubset(train_labels), 'val labels must be subset of train labels'
    train_labels = list(train_labels)
    val_labels = list(val_labels)
    label2id = {label: idx for idx, label in enumerate(train_labels)}
    id2label = {idx: label for idx, label in enumerate(train_labels)}

    # define coompute metrics function
    return_entity_level_metrics = False
    metric = evaluate.load("seqeval")
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "eval_precision": results["overall_precision"],
                "eval_recall": results["overall_recall"],
                "eval_f1": results["overall_f1"],
                "eval_accuracy": results["overall_accuracy"],
            }

    # load model and processor
    model, processor = get_model_from_config(config, label2id=label2id, id2label=id2label)
    print(f'Using model: {model.__class__.__name__}')
    print(f'Using processor: {processor.__class__.__name__}')

    # pdb.set_trace()
    # get data loader
    train_file_paths = get_file_paths(config.data.train_dir, ls_exclude_dir=config.data.ls_exclude_dir)
    val_file_paths = get_file_paths(config.data.val_dir, ls_exclude_dir=config.data.ls_exclude_dir)
    training_data_args = {k: v for k, v in vars(config.training.data_args).items() if not k.startswith('__')}
    train_dataset = TestCORDDataset(
        mode='train',
        file_paths=train_file_paths,
        processor=processor,
        label2id=label2id,
        model_type=config.model.model_type,
        max_length=512,
        **training_data_args
    )
    val_data_args = {k: v for k, v in vars(config.validation.data_args).items() if not k.startswith('__')}
    val_dataset = TestCORDDataset(
        mode='val',
        file_paths=val_file_paths,
        processor=processor,
        label2id=label2id,
        model_type=config.model.model_type,
        max_length=512,
        **val_data_args
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=8)
    print(f'Num train samples: {len(train_dataset)}')
    check_data_loader(train_loader)
    print(f'Num val samples: {len(val_dataset)}')
    check_data_loader(val_loader)
    pdb.set_trace()


    # train
    trainer_args = {k: v for k, v in vars(config.training.trainer_args).items() if not k.startswith('__')}
    training_args = TrainingArguments(**trainer_args)
    class CustomTrainer(Trainer):
        def get_train_dataloader(self):
            return train_loader

        def get_eval_dataloader(self, eval_dataset = None):
            return val_loader

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print('Training done, best ckpt path: ', trainer.state.best_model_checkpoint)
    with open(os.path.join(config.training.trainer_args.output_dir, 'best_model_ckpt.txt'), 'w') as f:
        f.write(str(trainer.state.best_model_checkpoint))