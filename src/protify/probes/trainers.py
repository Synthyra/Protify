import torch
import os
import numpy as np
from typing import Optional
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from dataclasses import dataclass
from probes.hybrid_probe import HybridProbe, HybridProbeConfig
from data.dataset_classes import (
    EmbedsLabelsDatasetFromDisk,
    PairEmbedsLabelsDatasetFromDisk,
    EmbedsLabelsDataset,
    PairEmbedsLabelsDataset,
    StringLabelDataset,
    PairStringLabelDataset,
)
from data.data_collators import (
    EmbedsLabelsCollator,
    PairEmbedsLabelsCollator,
    PairCollator_input_ids,
    StringLabelsCollator,
)
from visualization.ci_plots import regression_ci_plot, classification_ci_plot
from utils import print_message
from metrics import get_compute_metrics


@dataclass
class TrainerArguments:
    def __init__(
            self,
            model_save_dir: str,
            num_epochs: int = 200,
            probe_batch_size: int = 64,
            base_batch_size: int = 4,
            probe_grad_accum: int = 1,
            base_grad_accum: int = 1,
            lr: float = 1e-4,
            weight_decay: float = 0.00,
            task_type: str = 'regression',
            patience: int = 3,
            read_scaler: int = 100,
            save_model: bool = False,
            seed: int = 42,
            train_data_size: int = 100,
            plots_dir: str = None,
            full_finetuning: bool = False,
            hybrid_probe: bool = False,
            num_workers: int = 0,
            **kwargs
    ):
        self.model_save_dir = model_save_dir
        self.num_epochs = num_epochs
        self.probe_batch_size = probe_batch_size
        self.base_batch_size = base_batch_size
        self.probe_grad_accum = probe_grad_accum
        self.base_grad_accum = base_grad_accum
        self.lr = lr
        self.weight_decay = weight_decay
        self.task_type = task_type
        self.patience = patience
        self.save = save_model
        self.read_scaler = read_scaler
        self.seed = seed
        self.train_data_size = train_data_size
        self.plots_dir = plots_dir
        self.full_finetuning = full_finetuning
        self.hybrid_probe = hybrid_probe
        self.num_workers = num_workers

    def __call__(self, probe: Optional[bool] = True):
        if self.train_data_size > 100000:
            eval_strats = {
                'eval_strategy': 'steps',
                'eval_steps': 5000,
                'save_strategy': 'steps',
                'save_steps': 5000,
            }
        else:
            eval_strats = {
                'eval_strategy': 'epoch',
                'save_strategy': 'epoch',
            }

        if '/' in self.model_save_dir:
            save_dir = self.model_save_dir.split('/')[-1]
        else:
            save_dir = self.model_save_dir

        batch_size = self.probe_batch_size if probe else self.base_batch_size
        grad_accum = self.probe_grad_accum if probe else self.base_grad_accum
        warmup_steps = 100 if probe else 1000
        return TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=float(self.lr),
            lr_scheduler_type='cosine',
            weight_decay=float(self.weight_decay),
            warmup_steps=warmup_steps,
            save_total_limit=3,
            logging_steps=1000,
            report_to='none',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            seed=self.seed,
            label_names=['labels'],
            dataloader_num_workers=self.num_workers,
            dataloader_prefetch_factor=2 if self.num_workers > 0 else None,
            **eval_strats
        )


class TrainerMixin:
    def __init__(self, trainer_args: Optional[TrainerArguments] = None):
        self.trainer_args = trainer_args

    def _train(
            self,
            model,
            train_dataset,
            valid_dataset,
            test_dataset,
            data_collator,
            log_id,
            model_name,
            data_name,
            probe: Optional[bool] = True,
        ):
        task_type = self.trainer_args.task_type
        compute_metrics = get_compute_metrics(task_type)
        self.trainer_args.train_data_size = len(train_dataset)
        hf_trainer_args = self.trainer_args(probe=probe)
        ### TODO add options for optimizers and schedulers
        trainer = Trainer(
            model=model,
            args=hf_trainer_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.trainer_args.patience)]
        )
        trainer.can_return_loss = True
        metrics = trainer.evaluate(test_dataset)
        print_message(f'Initial metrics: {metrics}')

        trainer.train()

        valid_metrics = trainer.evaluate(valid_dataset)
        print_message(f'Final validation metrics: {valid_metrics}')

        y_pred, y_true, test_metrics = trainer.predict(test_dataset)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        if isinstance(y_true, tuple):
            y_true = y_true[0]

        y_pred, y_true = y_pred.astype(np.float32), y_true.astype(np.float32)
        print_message(f'y_pred: {y_pred.shape}\ny_true: {y_true.shape}\nFinal test metrics: \n{test_metrics}\n')

        output_dir = os.path.join(self.trainer_args.plots_dir, log_id)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}.png")
        title = f"{data_name} {model_name} {log_id}"

        if task_type == 'regression':
            regression_ci_plot(y_true, y_pred, save_path, title)
        else:
            classification_ci_plot(y_true, y_pred, save_path, title)

        if self.trainer_args.save:
            try:
                hub_path = os.path.join(self.full_args.hf_username, f"{data_name}_{model_name}_{log_id}")
                trainer.model.push_to_hub(hub_path, private=True)
            except Exception as e:
                print_message(f'Error saving model: {e}')

        model = trainer.model.cpu()
        trainer.accelerator.free_memory()
        torch.cuda.empty_cache()
        return model, valid_metrics, test_metrics

    def trainer_probe(
            self,
            model,
            tokenizer,
            model_name,
            data_name,
            train_dataset,
            valid_dataset,
            test_dataset,
            emb_dict=None,
            ppi=False,
            log_id=None,
        ):
        batch_size = self.trainer_args.probe_batch_size
        read_scaler = self.trainer_args.read_scaler
        input_dim = self.probe_args.input_dim
        task_type = self.probe_args.task_type
        full = self.embedding_args.matrix_embed
        db_path = os.path.join(self.embedding_args.embedding_save_dir, f'{model_name}_{full}.db')

        if self.embedding_args.sql:
            print('SQL enabled')
            if ppi:
                if full:
                    raise ValueError('Full matrix embeddings not currently supported for SQL and PPI') # TODO: Implement
                DatasetClass = PairEmbedsLabelsDatasetFromDisk
                CollatorClass = PairEmbedsLabelsCollator
            else:
                DatasetClass = EmbedsLabelsDatasetFromDisk
                CollatorClass = EmbedsLabelsCollator
        else:
            print('SQL disabled')
            if ppi:
                DatasetClass = PairEmbedsLabelsDataset
                CollatorClass = PairEmbedsLabelsCollator
            else:
                DatasetClass = EmbedsLabelsDataset
                CollatorClass = EmbedsLabelsCollator

        """
        For collator need to pass tokenizer, full, task_type
        For dataset need to pass
        hf_dataset, col_a, col_b, label_col, input_dim, task_type, db_path, emb_dict, batch_size, read_scaler, full, train
        """

        data_collator = CollatorClass(tokenizer=tokenizer, full=full, task_type=task_type)
        train_dataset = DatasetClass(
            hf_dataset=train_dataset,
            input_dim=input_dim,
            task_type=task_type,
            db_path=db_path,
            emb_dict=emb_dict,
            batch_size=batch_size,
            read_scaler=read_scaler,
            full=full,
            train=True
        )
        valid_dataset = DatasetClass(
            hf_dataset=valid_dataset,
            input_dim=input_dim,
            task_type=task_type,
            db_path=db_path,
            emb_dict=emb_dict,
            batch_size=batch_size,
            read_scaler=read_scaler,
            full=full,
            train=False
        )
        test_dataset = DatasetClass(
            hf_dataset=test_dataset,
            input_dim=input_dim,
            task_type=task_type,
            db_path=db_path,
            emb_dict=emb_dict,
            batch_size=batch_size,
            read_scaler=read_scaler,
            full=full,
            train=False
        )
        return self._train(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            data_collator=data_collator,
            log_id=log_id,
            model_name=model_name,
            data_name=data_name,
            probe=True,
        )

    def trainer_base_model(
            self,
            model,
            tokenizer,
            model_name,
            data_name,
            train_dataset,
            valid_dataset,
            test_dataset,
            ppi=False,
            log_id=None,
        ):
        task_type = self.probe_args.task_type

        if ppi:
            DatasetClass = PairStringLabelDataset
            CollatorClass = PairCollator_input_ids
        else:
            DatasetClass = StringLabelDataset
            CollatorClass = StringLabelsCollator

        data_collator = CollatorClass(tokenizer=tokenizer, task_type=task_type)

        train_dataset = DatasetClass(hf_dataset=train_dataset, train=True)
        valid_dataset = DatasetClass(hf_dataset=valid_dataset, train=False)
        test_dataset = DatasetClass(hf_dataset=test_dataset, train=False)

        return self._train(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            data_collator=data_collator,
            log_id=log_id,
            model_name=model_name,
            data_name=data_name,
            probe=False,
        )

    def trainer_hybrid_model(
            self,
            model,
            tokenizer,
            probe,
            model_name,
            data_name,
            train_dataset,
            valid_dataset,
            test_dataset,
            emb_dict=None,
            ppi=False,
            log_id=None,
        ):
            probe, _, _ = self.trainer_probe(
                model=probe,
                tokenizer=tokenizer,
                model_name=model_name,
                data_name=data_name,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                test_dataset=test_dataset,
                emb_dict=emb_dict,
                ppi=ppi,
                log_id=log_id,
            )
            config = HybridProbeConfig(
                tokenwise=self.probe_args.tokenwise,
                matrix_embed=self.embedding_args.matrix_embed,
                pooling_types=self.embedding_args.pooling_types,
            )

            hybrid_model = HybridProbe(config=config, model=model, probe=probe)
            
            return self.trainer_base_model(
                model=hybrid_model,
                tokenizer=tokenizer,
                model_name=model_name,
                data_name=data_name,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                test_dataset=test_dataset,
                ppi=ppi,
                log_id=log_id,
            )






