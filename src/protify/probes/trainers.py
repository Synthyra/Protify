import torch
import os
import numpy as np
from copy import deepcopy
from typing import Optional, Dict, List, Any, Union
from huggingface_hub import HfApi
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from dataclasses import dataclass
try:
    from probes.hybrid_probe import HybridProbe, HybridProbeConfig
    from probes.export_packaged_model import export_packaged_model_to_hub
    from data.dataset_classes import (
        EmbedsLabelsDatasetFromDisk,
        PairEmbedsLabelsDatasetFromDisk,
        EmbedsLabelsDataset,
        PairEmbedsLabelsDataset,
        StringLabelDataset,
        PairStringLabelDataset,
        MultiEmbedsLabelsDatasetFromDisk,
        MultiEmbedsLabelsDataset,
    )
except ImportError:
    from .hybrid_probe import HybridProbe, HybridProbeConfig
    from .export_packaged_model import export_packaged_model_to_hub
    from ..data.dataset_classes import (
        EmbedsLabelsDatasetFromDisk,
        PairEmbedsLabelsDatasetFromDisk,
        EmbedsLabelsDataset,
        PairEmbedsLabelsDataset,
        StringLabelDataset,
        PairStringLabelDataset,
        MultiEmbedsLabelsDatasetFromDisk,
        MultiEmbedsLabelsDataset,
    )
try:
    from data.data_collators import (
        EmbedsLabelsCollator,
        PairEmbedsLabelsCollator,
        PairCollator_input_ids,
        StringLabelsCollator,
    )
    from visualization.ci_plots import regression_ci_plot, classification_ci_plot
    from utils import print_message
    from metrics import get_compute_metrics
    from seed_utils import set_global_seed
    from probes.get_probe import get_probe
except ImportError:
    from ..data.data_collators import (
        EmbedsLabelsCollator,
        PairEmbedsLabelsCollator,
        PairCollator_input_ids,
        StringLabelsCollator,
    )
    from ..visualization.ci_plots import regression_ci_plot, classification_ci_plot
    from ..utils import print_message
    from ..metrics import get_compute_metrics
    from ..seed_utils import set_global_seed
    from .get_probe import get_probe


def _compute_eval_accumulation_steps(
    eval_dataset_size: int,
    batch_size: int,
    num_labels: int,
    task_type: str,
) -> Optional[int]:
    """Compute eval_accumulation_steps based on prediction size vs GPU memory.

    Returns None when all predictions fit comfortably on GPU (fastest path).
    Otherwise returns a step count that keeps accumulated predictions under
    5% of total GPU memory.
    """
    if task_type == 'regression':
        output_dim = 1
    elif task_type == 'binary':
        output_dim = 2
    else:
        output_dim = num_labels

    total_pred_bytes = eval_dataset_size * output_dim * 4  # float32

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_mem_bytes = props.total_memory
    else:
        gpu_mem_bytes = 8 * (1024 ** 3)  # 8 GB fallback

    budget_bytes = gpu_mem_bytes * 0.05

    if total_pred_bytes <= budget_bytes:
        return None

    bytes_per_step = batch_size * output_dim * 4
    return max(1, int(budget_bytes / bytes_per_step))


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
            push_raw_probe: bool = False,
            push_raw_probe_repo: str = None,
            seed: int = 42,
            train_data_size: int = 100,
            plots_dir: str = None,
            full_finetuning: bool = False,
            hybrid_probe: bool = False,
            num_workers: int = 0,
            make_plots: bool = True,
            num_runs: int = 1,
            torch_compile: bool = True,
            eval_accumulation_steps: Union[int, str] = "auto",
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
        self.push_raw_probe = push_raw_probe
        self.push_raw_probe_repo = push_raw_probe_repo
        self.read_scaler = read_scaler
        self.seed = seed
        self.train_data_size = train_data_size
        self.plots_dir = plots_dir
        self.full_finetuning = full_finetuning
        self.hybrid_probe = hybrid_probe
        self.num_workers = num_workers
        self.make_plots = make_plots
        self.num_runs = num_runs
        self.torch_compile = torch_compile
        self.eval_accumulation_steps = eval_accumulation_steps

    def __call__(self, probe: Optional[bool] = True):
        batch_size = self.probe_batch_size if probe else self.base_batch_size
        grad_accum = self.probe_grad_accum if probe else self.base_grad_accum

        if self.train_data_size > 250000:
            eval_steps = max(1, int(100000 / (batch_size * grad_accum)))
            eval_strats = {
                'eval_strategy': 'steps',
                'eval_steps': eval_steps,
                'save_strategy': 'steps',
                'save_steps': eval_steps,
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

        warmup_steps = 100 if probe else 1000

        if self.eval_accumulation_steps == "auto":
            eval_accum = _compute_eval_accumulation_steps(
                eval_dataset_size=getattr(self, 'eval_dataset_size', 10000),
                batch_size=batch_size,
                num_labels=getattr(self, 'num_labels', 2),
                task_type=self.task_type,
            )
        else:
            eval_accum = self.eval_accumulation_steps

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
            # Explicitly disable mixed precision training to prevent automatic fp16 conversion
            fp16=False,
            bf16=False,
            torch_compile=self.torch_compile,
            eval_accumulation_steps=eval_accum,
            **eval_strats
        )


class TrainerMixin:
    def __init__(self, trainer_args: Optional[TrainerArguments] = None):
        self.trainer_args = trainer_args

    def _format_metric_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    def _format_metrics_markdown(self, metrics: Dict[str, Any]) -> str:
        if metrics is None or len(metrics) == 0:
            return "- No metrics recorded."
        lines = []
        for key in sorted(metrics.keys()):
            lines.append(f"- `{key}`: {self._format_metric_value(metrics[key])}")
        return "\n".join(lines)

    def _build_model_card(
            self,
            repo_id: str,
            data_name: str,
            model_name: str,
            log_id: str,
            train_dataset,
            valid_dataset,
            test_dataset,
            valid_metrics: Dict[str, Any],
            test_metrics: Dict[str, Any],
        ) -> str:
        train_size = len(train_dataset)
        valid_size = "N/A" if valid_dataset is None else str(len(valid_dataset))
        test_size = len(test_dataset)
        task_type = self.trainer_args.task_type
        num_runs = self.trainer_args.num_runs
        validation_metrics_text = self._format_metrics_markdown(valid_metrics)
        test_metrics_text = self._format_metrics_markdown(test_metrics)
        return f"""---
library_name: transformers
tags: []
---

# {repo_id}

Fine-tuned with Protify.

## About Protify

Protify is an open source platform designed to simplify and democratize workflows for chemical language models. With Protify, deep learning models can be trained to predict chemical properties without requiring extensive coding knowledge or computational resources.

### Why Protify?

- Benchmark multiple models efficiently.
- Flexible for all skill levels.
- Accessible computing with support for precomputed embeddings.
- Cost-effective workflows for training and evaluation.

## Training Run

- `dataset`: {data_name}
- `model`: {model_name}
- `run_id`: {log_id}
- `task_type`: {task_type}
- `num_runs`: {num_runs}

## Dataset Statistics

- `train_size`: {train_size}
- `valid_size`: {valid_size}
- `test_size`: {test_size}

## Validation Metrics

{validation_metrics_text}

## Test Metrics

{test_metrics_text}
"""

    def _train(
            self,
            model,
            train_dataset,
            valid_dataset,
            test_dataset,
            data_collator,
            tokenizer,
            log_id,
            model_name,
            data_name,
            source_model_name: Optional[str] = None,
            ppi: bool = False,
            probe: Optional[bool] = True,
            skip_plot: bool = False,
        ):
        task_type = self.trainer_args.task_type
        tokenwise = self.probe_args.tokenwise
        compute_metrics = get_compute_metrics(task_type, tokenwise=tokenwise)
        self.trainer_args.train_data_size = len(train_dataset)
        self.trainer_args.num_labels = self.probe_args.num_labels
        self.trainer_args.eval_dataset_size = len(valid_dataset) if valid_dataset is not None else len(test_dataset)
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

        train_output = trainer.train()
        train_runtime = train_output.metrics.get('train_runtime', 0.0)

        valid_metrics = trainer.evaluate(valid_dataset)
        print_message(f'Final validation metrics: {valid_metrics}')

        y_pred, y_true, test_metrics = trainer.predict(test_dataset)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        if isinstance(y_true, tuple):
            y_true = y_true[0]

        y_pred, y_true = y_pred.astype(np.float32), y_true.astype(np.float32)
        
        # Remove singleton dimension if present
        if y_pred.ndim == 3 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(1)
        if y_true.ndim == 3 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)
        
        test_metrics['training_time_seconds'] = train_runtime
        print_message(f'y_pred: {y_pred.shape}\ny_true: {y_true.shape}\nFinal test metrics: \n{test_metrics}\n')

        if self.trainer_args.make_plots and self.trainer_args.plots_dir is not None and not skip_plot:
            output_dir = os.path.join(self.trainer_args.plots_dir, log_id)
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}.png")
            title = f"{data_name} {model_name} {log_id}"

            if task_type == 'regression':
                regression_ci_plot(y_true, y_pred, save_path, title)
            else:
                classification_ci_plot(y_true, y_pred, save_path, title)

        if source_model_name is None:
            source_model_name = model_name

        if self.trainer_args.save:
            try:
                hf_username = self.full_args.hf_username
                if hf_username is None or hf_username == "":
                    print_message("Warning: hf_username is not set. Cannot save model to HuggingFace Hub.")
                else:
                    repo_id = self.trainer_args.push_raw_probe_repo or f"{hf_username}/{data_name}_{model_name}_{log_id}"
                    hf_token = self.full_args.hf_token
                    if hf_token is None:
                        hf_token = os.environ.get("HF_TOKEN")

                    model_card = self._build_model_card(
                        repo_id=repo_id,
                        data_name=data_name,
                        model_name=model_name,
                        log_id=log_id,
                        train_dataset=train_dataset,
                        valid_dataset=valid_dataset,
                        test_dataset=test_dataset,
                        valid_metrics=valid_metrics,
                        test_metrics=test_metrics,
                    )

                    if self.trainer_args.push_raw_probe and (probe or isinstance(trainer.model, HybridProbe)):
                        probe_to_push = trainer.model.probe if isinstance(trainer.model, HybridProbe) else trainer.model
                        if hf_token is not None:
                            probe_to_push.push_to_hub(repo_id, private=True, token=hf_token)
                            api = HfApi(token=hf_token)
                        else:
                            probe_to_push.push_to_hub(repo_id, private=True)
                            api = HfApi()
                        api.upload_file(
                            path_or_fileobj=model_card.encode("utf-8"),
                            path_in_repo="README.md",
                            repo_id=repo_id,
                            repo_type="model",
                        )
                        print_message(f"Raw probe uploaded to Hugging Face Hub: {repo_id} (load with e.g. Class.from_pretrained('{repo_id}'))")
                    else:
                        packaged_export_succeeded = False
                        if probe or isinstance(trainer.model, HybridProbe):
                            try:
                                packaged_export_succeeded, export_message = export_packaged_model_to_hub(
                                    trained_model=trainer.model,
                                    source_model_name=source_model_name,
                                    probe_args=self.probe_args,
                                    embedding_args=self.embedding_args,
                                    tokenizer=tokenizer,
                                    repo_id=repo_id,
                                    model_card=model_card,
                                    ppi=ppi,
                                    private=True,
                                    hf_token=hf_token,
                                )
                                print_message(export_message)
                            except Exception as packaged_error:
                                print_message(f"Warning: packaged export failed for {repo_id}: {packaged_error}")

                        if not packaged_export_succeeded:
                            print_message(f"Falling back to direct model push_to_hub for {repo_id}")
                            if hf_token is not None:
                                trainer.model.push_to_hub(repo_id, private=True, token=hf_token)
                                api = HfApi(token=hf_token)
                            else:
                                trainer.model.push_to_hub(repo_id, private=True)
                                api = HfApi()
                            api.upload_file(
                                path_or_fileobj=model_card.encode("utf-8"),
                                path_in_repo="README.md",
                                repo_id=repo_id,
                                repo_type="model",
                            )
                    print_message(f"Successfully saved model to HuggingFace Hub: {repo_id}")
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print_message(f"Error saving model to HuggingFace Hub: {e}")
                print_message(f"Error traceback: {error_trace}")
                print_message(f"save_model flag: {self.trainer_args.save}")

        model = trainer.model.cpu()
        trainer.accelerator.free_memory()
        torch.cuda.empty_cache()
        return model, valid_metrics, test_metrics, y_pred, y_true

    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across multiple runs, computing mean ± std for each metric."""
        if not metrics_list:
            return {}
        
        # Collect all metric keys
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        aggregated = {}
        for key in all_keys:
            values = [m.get(key) for m in metrics_list if key in m and m[key] is not None]
            if not values:
                continue
            
            # Check if all values are numeric
            if all(isinstance(v, (int, float)) for v in values):
                mean_val = np.mean(values)
                std_val = np.std(values)
                # Store as formatted string with mean±std
                aggregated[key] = f"{mean_val:.4f}±{std_val:.4f}"
                # Also store raw mean for sorting/comparison purposes
                aggregated[f"{key}_mean"] = float(mean_val)
                aggregated[f"{key}_std"] = float(std_val)
            else:
                # For non-numeric values, just take the first one
                aggregated[key] = values[0]
        
        return aggregated

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
            skip_plot=False,
            source_model_name: Optional[str] = None,
        ):
        batch_size = self.trainer_args.probe_batch_size
        read_scaler = self.trainer_args.read_scaler
        input_size = self.probe_args.input_size
        task_type = self.probe_args.task_type
        tokenwise = self.probe_args.tokenwise
        num_runs = getattr(self.trainer_args, 'num_runs', 1)
        base_seed = self.trainer_args.seed
        
        print(f'task_type: {task_type}')
        full = self.embedding_args.matrix_embed
        db_path = os.path.join(self.embedding_args.embedding_save_dir, f'{model_name}_{full}.db')

        use_multi = getattr(self.full_args, 'multi_column', None)
        if self.embedding_args.sql:
            print('SQL enabled')
            if ppi:
                DatasetClass = PairEmbedsLabelsDatasetFromDisk
                CollatorClass = PairEmbedsLabelsCollator
            elif use_multi:
                DatasetClass = MultiEmbedsLabelsDatasetFromDisk
                CollatorClass = EmbedsLabelsCollator
            else:
                DatasetClass = EmbedsLabelsDatasetFromDisk
                CollatorClass = EmbedsLabelsCollator
        else:
            print('SQL disabled')
            if ppi:
                DatasetClass = PairEmbedsLabelsDataset
                CollatorClass = PairEmbedsLabelsCollator
            elif use_multi:
                DatasetClass = MultiEmbedsLabelsDataset
                CollatorClass = EmbedsLabelsCollator
            else:
                DatasetClass = EmbedsLabelsDataset
                CollatorClass = EmbedsLabelsCollator

        """
        For collator need to pass tokenizer, full, task_type
        For dataset need to pass
        hf_dataset, col_a, col_b, label_col, input_size, task_type, db_path, emb_dict, batch_size, read_scaler, full, train
        """

        add_token_ids = getattr(self.probe_args, 'add_token_ids', False)
        padding = getattr(self.full_args, 'padding', 'max_length')
        max_length = getattr(self.full_args, 'max_length', 2048)
        data_collator = CollatorClass(tokenizer=tokenizer, full=full, task_type=task_type, tokenwise=tokenwise, add_token_ids=add_token_ids, padding=padding, max_length=max_length)
        common_kwargs = dict(
            hf_dataset=train_dataset,
            input_size=input_size,
            task_type=task_type,
            db_path=db_path,
            emb_dict=emb_dict,
            batch_size=batch_size,
            read_scaler=read_scaler,
            full=full,
            train=True,
            random_pair_flipping=self.full_args.random_pair_flipping,
        )
        if use_multi:
            train_ds = DatasetClass(seq_cols=use_multi, **deepcopy(common_kwargs))
        else:
            train_ds = DatasetClass(**deepcopy(common_kwargs))
        
        # BUG FIX: Update hf_dataset in common_kwargs before creating validation and test datasets.
        # Previously, common_kwargs['hf_dataset'] was set to train_dataset and never updated,
        # causing valid_dataset and test_dataset to incorrectly use training data. This resulted
        # in valid_metrics and test_metrics being identical since they were computed on the same
        # (training) dataset. The fix ensures each dataset uses the correct HuggingFace dataset.
        # We use deepcopy to ensure each dataset gets an independent copy of the kwargs dictionary
        # to prevent any potential shared state issues.
        common_kwargs['train'] = False
        common_kwargs['hf_dataset'] = valid_dataset
        if use_multi:
            valid_ds = DatasetClass(seq_cols=use_multi, **deepcopy(common_kwargs))
        else:
            valid_ds = DatasetClass(**deepcopy(common_kwargs))
        common_kwargs['hf_dataset'] = test_dataset
        if use_multi:
            test_ds = DatasetClass(seq_cols=use_multi, **deepcopy(common_kwargs))
        else:
            test_ds = DatasetClass(**deepcopy(common_kwargs))
        
        # Single run - original behavior
        if num_runs == 1:
            return self._train(
                model=model,
                train_dataset=train_ds,
                valid_dataset=valid_ds,
                test_dataset=test_ds,
                data_collator=data_collator,
                tokenizer=tokenizer,
                log_id=log_id,
                model_name=model_name,
                data_name=data_name,
                source_model_name=source_model_name,
                ppi=ppi,
                probe=True,
                skip_plot=skip_plot,
            )
        
        # Multi-run mode: train multiple times with different seeds, reusing datasets
        print_message(f"Running {num_runs} training runs with different seeds for {data_name}/{model_name}")
        
        all_valid_metrics = []
        all_test_metrics = []
        run_results = []  # Store (run_idx, test_loss, y_pred, y_true, seed, model) for plotting best
        
        for run_idx in range(num_runs):
            run_seed = base_seed + run_idx
            self.trainer_args.seed = run_seed
            set_global_seed(run_seed)
            
            print_message(f"=== Run {run_idx + 1}/{num_runs} with seed {run_seed} ===")
            
            # Create a fresh probe for each run
            probe = get_probe(self.probe_args)
            
            run_model, valid_metrics, test_metrics, y_pred, y_true = self._train(
                model=probe,
                train_dataset=train_ds,
                valid_dataset=valid_ds,
                test_dataset=test_ds,
                data_collator=data_collator,
                tokenizer=tokenizer,
                log_id=f"{log_id}_run{run_idx}",
                model_name=model_name,
                data_name=data_name,
                source_model_name=source_model_name,
                ppi=ppi,
                probe=True,
                skip_plot=True,  # Skip plots during individual runs
            )
            
            all_valid_metrics.append(valid_metrics)
            all_test_metrics.append(test_metrics)
            
            # Track test loss for determining best run
            test_loss = test_metrics.get('test_loss', test_metrics.get('eval_loss', float('inf')))
            run_results.append((run_idx, test_loss, y_pred, y_true, run_seed, run_model))
        
        # Restore original seed
        self.trainer_args.seed = base_seed
        
        # Compute aggregated metrics (mean ± std)
        aggregated_valid = self._aggregate_metrics(all_valid_metrics)
        aggregated_test = self._aggregate_metrics(all_test_metrics)
        
        # Find the best run (lowest test loss)
        best_run = min(run_results, key=lambda x: x[1])
        best_run_idx, best_loss, best_y_pred, best_y_true, best_seed, best_model = best_run
        print_message(f"Best run: {best_run_idx + 1} (seed={best_seed}, test_loss={best_loss:.4f})")
        
        # Generate plot for best run (unless skip_plot is True)
        if not skip_plot:
            output_dir = os.path.join(self.trainer_args.plots_dir, log_id)
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}_best.png")
            title = f"{data_name} {model_name} (best of {num_runs} runs, seed={best_seed})"
            
            if task_type == 'regression':
                regression_ci_plot(best_y_true, best_y_pred, save_path, title)
            else:
                classification_ci_plot(best_y_true, best_y_pred, save_path, title)
        
        # Return the best model along with aggregated metrics
        return best_model, aggregated_valid, aggregated_test, best_y_pred, best_y_true

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
            skip_plot=False,
            model_factory=None,
            source_model_name: Optional[str] = None,
        ):
        task_type = self.probe_args.task_type
        tokenwise = self.probe_args.tokenwise
        num_runs = getattr(self.trainer_args, 'num_runs', 1)
        base_seed = self.trainer_args.seed

        if ppi:
            DatasetClass = PairStringLabelDataset
            CollatorClass = PairCollator_input_ids
        else:
            DatasetClass = StringLabelDataset
            CollatorClass = StringLabelsCollator

        padding = getattr(self.full_args, 'padding', 'max_length')
        max_length = getattr(self.full_args, 'max_length', 2048)
        data_collator = CollatorClass(tokenizer=tokenizer, task_type=task_type, tokenwise=tokenwise, padding=padding, max_length=max_length)

        train_ds = DatasetClass(hf_dataset=train_dataset, train=True, random_pair_flipping=self.full_args.random_pair_flipping)
        valid_ds = DatasetClass(hf_dataset=valid_dataset, train=False, random_pair_flipping=self.full_args.random_pair_flipping)
        test_ds = DatasetClass(hf_dataset=test_dataset, train=False, random_pair_flipping=self.full_args.random_pair_flipping)

        # Single run - original behavior
        if num_runs == 1:
            return self._train(
                model=model,
                train_dataset=train_ds,
                valid_dataset=valid_ds,
                test_dataset=test_ds,
                data_collator=data_collator,
                tokenizer=tokenizer,
                log_id=log_id,
                model_name=model_name,
                data_name=data_name,
                source_model_name=source_model_name,
                ppi=ppi,
                probe=False,
                skip_plot=skip_plot,
            )
        
        # Multi-run mode: train multiple times with different seeds
        print_message(f"Running {num_runs} full finetuning runs with different seeds for {data_name}/{model_name}")
        
        all_valid_metrics = []
        all_test_metrics = []
        run_results = []  # Store (run_idx, test_loss, y_pred, y_true, seed, model) for plotting best
        
        for run_idx in range(num_runs):
            run_seed = base_seed + run_idx
            self.trainer_args.seed = run_seed
            set_global_seed(run_seed)
            
            print_message(f"=== Run {run_idx + 1}/{num_runs} with seed {run_seed} ===")
            
            # Create a fresh model for each run using the factory
            if model_factory is not None:
                run_model = model_factory()
            
            trained_model, valid_metrics, test_metrics, y_pred, y_true = self._train(
                model=run_model,
                train_dataset=train_ds,
                valid_dataset=valid_ds,
                test_dataset=test_ds,
                data_collator=data_collator,
                tokenizer=tokenizer,
                log_id=f"{log_id}_run{run_idx}",
                model_name=model_name,
                data_name=data_name,
                source_model_name=source_model_name,
                ppi=ppi,
                probe=False,
                skip_plot=True,  # Skip plots during individual runs
            )
            
            all_valid_metrics.append(valid_metrics)
            all_test_metrics.append(test_metrics)
            
            # Track test loss for determining best run
            test_loss = test_metrics.get('test_loss', test_metrics.get('eval_loss', float('inf')))
            run_results.append((run_idx, test_loss, y_pred, y_true, run_seed, trained_model))
        
        # Restore original seed
        self.trainer_args.seed = base_seed
        
        # Compute aggregated metrics (mean ± std)
        aggregated_valid = self._aggregate_metrics(all_valid_metrics)
        aggregated_test = self._aggregate_metrics(all_test_metrics)
        
        # Find the best run (lowest test loss)
        best_run = min(run_results, key=lambda x: x[1])
        best_run_idx, best_loss, best_y_pred, best_y_true, best_seed, best_model = best_run
        print_message(f"Best run: {best_run_idx + 1} (seed={best_seed}, test_loss={best_loss:.4f})")
        
        # Generate plot for best run (unless skip_plot is True)
        if not skip_plot:
            output_dir = os.path.join(self.trainer_args.plots_dir, log_id)
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}_best.png")
            title = f"{data_name} {model_name} (best of {num_runs} runs, seed={best_seed})"
            
            if task_type == 'regression':
                regression_ci_plot(best_y_true, best_y_pred, save_path, title)
            else:
                classification_ci_plot(best_y_true, best_y_pred, save_path, title)
        
        # Return the best model along with aggregated metrics
        return best_model, aggregated_valid, aggregated_test, best_y_pred, best_y_true

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
            skip_plot=False,
            model_factory=None,
            probe_factory=None,
            source_model_name: Optional[str] = None,
        ):
            num_runs = getattr(self.trainer_args, 'num_runs', 1)
            base_seed = self.trainer_args.seed
            
            # Single run - original behavior
            if num_runs == 1:
                return self._train_hybrid_single_run(
                    model=model,
                    tokenizer=tokenizer,
                    probe=probe,
                    model_name=model_name,
                    data_name=data_name,
                    train_dataset=train_dataset,
                    valid_dataset=valid_dataset,
                    test_dataset=test_dataset,
                    emb_dict=emb_dict,
                    ppi=ppi,
                    log_id=log_id,
                    skip_plot=skip_plot,
                    source_model_name=source_model_name,
                )
            
            # Multi-run mode for hybrid probe
            # For hybrid probe, we only care about final metrics, not intermediate probe metrics
            # training_time_seconds should sum both probe and model+probe training times
            print_message(f"Running {num_runs} hybrid probe runs with different seeds for {data_name}/{model_name}")
            
            all_valid_metrics = []
            all_test_metrics = []
            run_results = []  # Store (run_idx, test_loss, y_pred, y_true, seed, model) for plotting best
            
            for run_idx in range(num_runs):
                run_seed = base_seed + run_idx
                self.trainer_args.seed = run_seed
                set_global_seed(run_seed)
                
                print_message(f"=== Hybrid Run {run_idx + 1}/{num_runs} with seed {run_seed} ===")
                
                # Create fresh probe and model for each run using factories
                if probe_factory is not None:
                    run_probe = probe_factory()
                if model_factory is not None:
                    run_model = model_factory()
                
                trained_model, valid_metrics, test_metrics, y_pred, y_true = self._train_hybrid_single_run(
                    model=run_model,
                    tokenizer=tokenizer,
                    probe=run_probe,
                    model_name=model_name,
                    data_name=data_name,
                    train_dataset=train_dataset,
                    valid_dataset=valid_dataset,
                    test_dataset=test_dataset,
                    emb_dict=emb_dict,
                    ppi=ppi,
                    log_id=f"{log_id}_run{run_idx}",
                    skip_plot=True,  # Skip plots during individual runs
                    source_model_name=source_model_name,
                )
                
                # Only collect final metrics (not intermediate probe metrics)
                all_valid_metrics.append(valid_metrics)
                all_test_metrics.append(test_metrics)
                
                # Track test loss for determining best run
                test_loss = test_metrics.get('test_loss', test_metrics.get('eval_loss', float('inf')))
                run_results.append((run_idx, test_loss, y_pred, y_true, run_seed, trained_model))
            
            # Restore original seed
            self.trainer_args.seed = base_seed
            
            # Compute aggregated metrics (mean ± std)
            # This will include training_time_seconds which already has probe + base time summed per run
            aggregated_valid = self._aggregate_metrics(all_valid_metrics)
            aggregated_test = self._aggregate_metrics(all_test_metrics)
            
            # Find the best run (lowest test loss)
            best_run = min(run_results, key=lambda x: x[1])
            best_run_idx, best_loss, best_y_pred, best_y_true, best_seed, best_model = best_run
            print_message(f"Best hybrid run: {best_run_idx + 1} (seed={best_seed}, test_loss={best_loss:.4f})")
            
            # Generate plot for best run (unless skip_plot is True)
            task_type = self.probe_args.task_type
            if not skip_plot:
                output_dir = os.path.join(self.trainer_args.plots_dir, log_id)
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}_best.png")
                title = f"{data_name} {model_name} hybrid (best of {num_runs} runs, seed={best_seed})"
                
                if task_type == 'regression':
                    regression_ci_plot(best_y_true, best_y_pred, save_path, title)
                else:
                    classification_ci_plot(best_y_true, best_y_pred, save_path, title)
            
            # Return the best model along with aggregated metrics
            return best_model, aggregated_valid, aggregated_test, best_y_pred, best_y_true

    def _train_hybrid_single_run(
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
            skip_plot=False,
            source_model_name: Optional[str] = None,
        ):
            """Single run of hybrid probe training (probe first, then model+probe)."""
            # Store original num_runs and temporarily set to 1 for the probe phase
            original_num_runs = getattr(self.trainer_args, 'num_runs', 1)
            self.trainer_args.num_runs = 1
            
            probe, _, probe_test_metrics, _, _ = self.trainer_probe(
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
                skip_plot=True,  # Always skip plot for probe phase in hybrid
                source_model_name=source_model_name,
            )
            
            # Restore num_runs
            self.trainer_args.num_runs = original_num_runs
            
            probe_time = probe_test_metrics.get('training_time_seconds')
            if not isinstance(probe_time, (int, float)):
                raise ValueError(f"Probe time is not a number: {probe_time}") # ensure we are capturing the time correctly
            config = HybridProbeConfig(
                tokenwise=self.probe_args.tokenwise,
                matrix_embed=self.embedding_args.matrix_embed,
                pooling_types=self.embedding_args.pooling_types,
            )

            hybrid_model = HybridProbe(config=config, model=model, probe=probe)

            # Temporarily set num_runs to 1 for the base model phase
            self.trainer_args.num_runs = 1
            
            base_model, base_valid_metrics, base_test_metrics, y_pred, y_true = self.trainer_base_model(
                model=hybrid_model,
                tokenizer=tokenizer,
                model_name=model_name,
                data_name=data_name,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                test_dataset=test_dataset,
                ppi=ppi,
                log_id=log_id,
                skip_plot=skip_plot,
                source_model_name=source_model_name,
            )
            
            # Restore num_runs
            self.trainer_args.num_runs = original_num_runs
            
            # Sum probe time and base time for total training time
            if probe_time is not None:
                base_time = base_test_metrics.get('training_time_seconds')
                if isinstance(base_time, (int, float)):
                    base_test_metrics['training_time_seconds'] = base_time + probe_time
                elif base_time is None:
                    base_test_metrics['training_time_seconds'] = probe_time
            return base_model, base_valid_metrics, base_test_metrics, y_pred, y_true