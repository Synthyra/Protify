from __future__ import annotations

import copy
import csv
import json
import math
import os
import yaml

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .base_models.get_base_models import get_tokenizer
from .console_output import print_message
from .embedder import Embedder
from .feature_cache import has_feature
# Imported through the package path, not bare. A bare `probes.*` import would load a
# second copy of every probe module under a different identity, and isinstance checks
# against the package-path classes would then silently fail.
from .probes.estimator_probe import is_estimator_probe
from .probes.get_probe import normalize_probe_type
from .run_tracking import curves_closed, job_summary, open_job, training_job


if TYPE_CHECKING:
    from .main import MainProcess  # main imports this module, so only the type checker reads main


# How a sweep's `method` searches, as Optuna samplers; a grid needs every parameter to list its values.
SWEEP_SAMPLERS = ("bayes", "grid", "random")


def suggest(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    """Draw one parameter of a trial from its W&B-format sweep entry."""
    if "value" in spec:
        return spec["value"]
    if "values" in spec:
        return trial.suggest_categorical(name, spec["values"])
    low, high = spec["min"], spec["max"]
    # W&B's default for a bare range: integers when both ends are, else uniform.
    distribution = spec.get("distribution", "int_uniform" if isinstance(low, int) and isinstance(high, int) else "uniform")
    if distribution == "uniform":
        return trial.suggest_float(name, low, high)
    if distribution == "log_uniform_values":
        return trial.suggest_float(name, low, high, log=True)
    if distribution == "int_uniform":
        return trial.suggest_int(name, low, high)
    if distribution == "q_uniform":
        return trial.suggest_float(name, low, high, step=spec.get("q", 1))
    if distribution == "q_log_uniform_values":
        q = spec.get("q", 1)
        return round(trial.suggest_float(name, low, high, log=True) / q) * q
    raise ValueError(f"Sweep parameter {name!r} has distribution {distribution!r}, which the sweep does not implement.")


def sweep_search(sweep: dict[str, Any], seed: int) -> Any:
    """The Optuna study that runs a sweep in this process: TPE for `bayes`, else random or grid search."""
    import optuna

    method = sweep["method"]
    if method not in SWEEP_SAMPLERS:
        raise ValueError(f"sweep method must be one of {SWEEP_SAMPLERS}; received {method!r}.")
    if method == "grid":
        ranged = sorted(name for name, spec in sweep["parameters"].items() if "value" not in spec and "values" not in spec)
        if ranged:
            raise ValueError(f"A grid sweep needs every parameter to list its values; {ranged} are ranges.")
        sampler = optuna.samplers.GridSampler(
            {name: spec.get("values", [spec.get("value")]) for name, spec in sweep["parameters"].items()}, seed=seed
        )
    else:
        sampler = optuna.samplers.TPESampler(seed=seed) if method == "bayes" else optuna.samplers.RandomSampler(seed=seed)
    return optuna.create_study(study_name=sweep.get("name"), sampler=sampler, direction=sweep["metric"]["goal"])


def _parse_pooling_types(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """Accept a list, a single token ('mean'), or a hyphen/comma-joined string ('mean-cls', 'mean,var')."""
    if isinstance(value, (list, tuple)):
        return list(value)
    assert isinstance(value, str), f'pooling types must be str or list, got {type(value)}'
    for sep in ('-', ','):
        if sep in value:
            return [tok.strip() for tok in value.split(sep) if tok.strip()]
    return [value.strip()]


def _probe_takes_pooling(probe_type: str, tokenwise: bool, pooling_types: list[str]) -> bool:
    """Whether this probe type takes these probe pooling types.

    Only the transformer sequence probe computes `bom`, and get_probe refuses it for any other.
    Lyra's classifier reads one pooled vector, so LyraConfig refuses more than one type. Every
    other probe type ignores the probe pooling.
    """
    if 'bom' in pooling_types and (probe_type != 'transformer' or tokenwise):
        return False

    return probe_type != 'lyra' or len(pooling_types) == 1


def _keep_values(spec: dict[str, Any], keep: Callable[[Any], bool]) -> dict[str, Any] | None:
    """Narrow a sweep entry that lists its `value` or `values` to the ones `keep` accepts.

    Returns None when no value is left.
    """
    if 'value' in spec:
        return spec if keep(spec['value']) else None

    kept = [index for index, value in enumerate(spec['values']) if keep(value)]
    if not kept:
        return None

    narrowed = {**spec, 'values': [spec['values'][index] for index in kept]}
    if 'probabilities' in spec:
        # wandb requires the probabilities of the values that are left to sum to one.
        weights = [spec['probabilities'][index] for index in kept]
        narrowed['probabilities'] = [weight / sum(weights) for weight in weights]
    return narrowed


def _pooling_every_probe_takes(
    spec: dict[str, Any],
    probe_types: list[str],
    tokenwise: bool,
) -> dict[str, Any] | None:
    """Narrow a `probe_pooling_types` sweep entry to the values that every one of `probe_types` takes.

    A trial draws its probe type and its pooling independently, so a value has to suit every probe
    type the sweep can pick. Returns None when no value is left.
    """
    def every_probe_takes(value: Any) -> bool:
        pooling_types = _parse_pooling_types(value)
        return all(_probe_takes_pooling(probe_type, tokenwise, pooling_types) for probe_type in probe_types)

    return _keep_values(spec, every_probe_takes)


# The bounded distributions a hidden size can be drawn from, each with its quantized form.
_QUANTIZED_DISTRIBUTIONS = {
    'int_uniform': 'q_uniform',
    'uniform': 'q_uniform',
    'q_uniform': 'q_uniform',
    'log_uniform_values': 'q_log_uniform_values',
    'q_log_uniform_values': 'q_log_uniform_values',
}


def _hidden_sizes_heads_divide(spec: dict[str, Any], head_size_spec: dict[str, Any]) -> dict[str, Any] | None:
    """Narrow a `hidden_size` sweep entry to multiples of every head size the sweep can draw.

    The transformer probe splits hidden_size into heads of head_size, and a trial draws the two
    independently. A range is quantized to the multiples inside it; wandb draws round(X / q) * q
    for X in [min, max], so bounds on multiples of q keep every draw in range. Returns the entry
    unchanged when either one comes from a distribution not handled here, and None when no
    multiple is left.
    """
    head_sizes = head_size_spec['values'] if 'values' in head_size_spec else [head_size_spec.get('value')]
    if None in head_sizes:
        return spec

    multiple = math.lcm(*(int(head_size) for head_size in head_sizes))
    if 'value' in spec or 'values' in spec:
        return _keep_values(spec, lambda hidden_size: int(hidden_size) % multiple == 0)

    # wandb reads a bare min and max as int_uniform or uniform, which quantize alike.
    quantized = _QUANTIZED_DISTRIBUTIONS.get(spec.get('distribution', 'uniform'))
    if quantized is None or 'min' not in spec or 'max' not in spec:
        return spec

    step = math.lcm(multiple, int(spec.get('q', 1)))
    low = math.ceil(spec['min'] / step) * step
    high = math.floor(spec['max'] / step) * step
    if low > high:
        return None

    if low == high:
        return {'value': low}

    return {**spec, 'distribution': quantized, 'min': low, 'max': high, 'q': step}


class HyperoptModule:
    def __init__(
        self, 
        main_process: MainProcess,
        model_name: str,
        data_name: str,
        dataset: tuple[Any, ...],
        emb_dict: Any,
        sweep_config: dict[str, Any],
        results_list: list[dict[str, Any]],
        swept_param_keys: list[str] = None,
        parameters: dict[str, dict[str, Any]] | None = None,
    ):
        self.mp = main_process
        self.model_name = model_name
        self.data_name = data_name
        self.dataset = dataset
        self.emb_dict = emb_dict
        self.sweep_config = sweep_config
        self.results_list = results_list
        self.swept_param_keys = swept_param_keys or []
        # The W&B-format entries each trial draws from, after narrowing to what the probe takes.
        self.parameters = parameters or {}
        
        self.base_probe_args = copy.deepcopy(self.mp.probe_args.__dict__)
        self.base_trainer_args = copy.deepcopy(self.mp.trainer_args.__dict__)
        self.base_embedding_args = copy.deepcopy(self.mp.embedding_args.__dict__)
        
        self.probe_keys = {
            'hidden_size','dropout','n_layers','pre_ln','classifier_size',
            'classifier_dropout','head_size','rotary','use_bias','probe_pooling_types',
            'lora','lora_r','lora_alpha','lora_dropout','probe_type','tokenwise', 'pooling_types',
            'add_token_ids','bom_k'
        }
        self.trainer_keys = {
            'lr','weight_decay','num_epochs','probe_batch_size',
            'base_batch_size','probe_grad_accum','base_grad_accum',
            'patience','seed'
        }
        self.full_args_keys = {
            'random_pair_flipping',
        }
        self.embedding_keys = {
            'embedding_pooling_types', 'embedding_hidden_state_index'
        }
        # Estimator hyperparameters do not live on ProbeArguments; they are constructor
        # arguments for the learner. A trial collects them into one dict that
        # MainProcess._estimator_overrides layers on top of --scikit_model_args.
        self.estimator_keys = {
            'learning_rate', 'max_depth', 'n_estimators', 'min_child_weight',
            'colsample_bytree', 'subsample', 'reg_lambda', 'reg_alpha', 'gamma',
            'num_leaves', 'max_bin', 'max_features', 'min_samples_leaf',
        }
        self.int_keys = {
            'hidden_size', 'n_layers', 'classifier_size', 'head_size',
            'lora_r', 'lora_alpha', 'num_epochs', 'probe_batch_size',
            'base_batch_size', 'probe_grad_accum', 'base_grad_accum',
            'patience', 'seed', 'embedding_hidden_state_index',
            'max_depth', 'n_estimators', 'min_child_weight', 'num_leaves',
            'min_samples_leaf', 'max_bin',
        }

    def apply_config(self, cfg: dict[str, Any]) -> None:
        self.mp.probe_args.__dict__.update(copy.deepcopy(self.base_probe_args))
        self.mp.trainer_args.__dict__.update(copy.deepcopy(self.base_trainer_args))
        self.mp.embedding_args.__dict__.update(copy.deepcopy(self.base_embedding_args))

        if 'n_heads' in cfg and 'head_size' not in cfg:
            import warnings
            warnings.warn(
                "'n_heads' in sweep config is deprecated; use 'head_size' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            hs = int(cfg['hidden_size']) if 'hidden_size' in cfg else self.mp.probe_args.hidden_size
            legacy = int(cfg.pop('n_heads'))
            assert hs % legacy == 0, f"hidden_size {hs} not divisible by legacy n_heads {legacy}"
            cfg['head_size'] = hs // legacy

        for key in self.int_keys:
            if key in cfg:
                cfg[key] = int(cfg[key])

        # Only the transformer probe splits hidden_size into attention heads. MLP, Lyra, and the
        # estimators have none, and an external probe's config checks its own heads.
        if 'hidden_size' in cfg:
            trial_probe_type = cfg.get('probe_type', self.mp.probe_args.probe_type)
            if normalize_probe_type(trial_probe_type) == 'transformer':
                hs = int(cfg['hidden_size'])
                head_size = int(cfg['head_size']) if 'head_size' in cfg else self.mp.probe_args.head_size
                assert hs % head_size == 0, (
                    f"sweep hidden_size {hs} not divisible by head_size {head_size}"
                )

        if 'dropout' in cfg:
            cfg['transformer_dropout'] = cfg['dropout']

        if 'probe_type' in cfg:
            # ProbeArguments normalizes in its constructor, which apply_config bypasses by
            # setting the attribute directly. A sweep carrying the older 'linear' name
            # would otherwise reach get_probe unnormalized.
            cfg['probe_type'] = normalize_probe_type(cfg['probe_type'])

        if 'probe_pooling_types' in cfg:
            cfg['probe_pooling_types'] = _parse_pooling_types(cfg['probe_pooling_types'])
            cfg['pooling_types'] = cfg['probe_pooling_types']
        if 'embedding_pooling_types' in cfg:
            cfg['embedding_pooling_types'] = _parse_pooling_types(cfg['embedding_pooling_types'])

        # A trial that sweeps probe_type sees both families' parameters, so start from an
        # empty override dict every time rather than carrying the previous trial's.
        self.mp.probe_args.estimator_overrides = {
            k: v for k, v in cfg.items() if k in self.estimator_keys
        }

        for k, v in cfg.items():
            if k in self.probe_keys and hasattr(self.mp.probe_args, k):
                setattr(self.mp.probe_args, k, v)
            if k in self.trainer_keys and hasattr(self.mp.trainer_args, k):
                setattr(self.mp.trainer_args, k, v)
            if k in self.full_args_keys and hasattr(self.mp.full_args, k):
                setattr(self.mp.full_args, k, v)
            # Handle embedding pooling types
            if k in self.embedding_keys:
                if k == 'embedding_pooling_types':
                    self.mp.embedding_args.pooling_types = v
                if k == 'embedding_hidden_state_index':
                    self.mp.embedding_args.hidden_state_index = v

    def reload_embeddings(self, ppi: bool) -> None:
        tokenizer = get_tokenizer(self.model_name)
        test_seq = self.mp.all_seqs[0]
        # A sweep can change the embedding pooling or hidden state between trials, and each setting is
        # its own store, so the store is looked up again for every trial.
        if self.mp._sql:
            if not has_feature(self.mp._embedding_store(self.model_name)):
                embedder = Embedder(self.mp.embedding_args, self.mp.all_seqs)
                embedder(self.model_name)
            input_dim = self.mp.get_embedding_dim_sql(self.mp._embedding_store(self.model_name), test_seq, tokenizer)
            self.emb_dict = None
        else:
            if not has_feature(self.mp._embedding_store(self.model_name)):
                previous_save_embeddings = self.mp.embedding_args.save_embeddings
                self.mp.embedding_args.save_embeddings = True
                embedder = Embedder(self.mp.embedding_args, self.mp.all_seqs)
                try:
                    embedder(self.model_name)
                finally:
                    self.mp.embedding_args.save_embeddings = previous_save_embeddings
            self.emb_dict = self.mp._embedding_dict(self.model_name, self.mp.all_seqs)
            input_dim = self.mp.get_embedding_dim_pth(self.emb_dict, test_seq, tokenizer)

        self.mp.probe_args.input_size = input_dim * 2 if (ppi and not self.mp._full) else input_dim
        self.mp.probe_args.max_seq_len = self.mp._max_length * 2 if (ppi and self.mp._full) else self.mp._max_length

    def train_model(self, sweep_mode: bool = True) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        train_set, valid_set, test_set, _, _, ppi = self.dataset
        
        if self.mp.full_args.full_finetuning:
             model, valid_metrics, test_metrics = self.mp._run_full_finetuning(
                self.model_name, self.data_name, 
                train_set, valid_set, test_set, 
                ppi=ppi, sweep_mode=sweep_mode
            )
             return model, valid_metrics, test_metrics

        elif self.mp.full_args.hybrid_probe:
            tokenizer = get_tokenizer(self.model_name)
            model, valid_metrics, test_metrics = self.mp._run_hybrid_probe(
                self.model_name, self.data_name,
                train_set, valid_set, test_set,
                tokenizer,
                emb_dict=self.emb_dict,
                ppi=ppi,
                sweep_mode=sweep_mode
            )
            return model, valid_metrics, test_metrics

        elif is_estimator_probe(self.mp.probe_args.probe_type):
            # The estimator path builds its own numpy matrices from the embedding cache,
            # so it needs the dataset rather than a tokenizer and an emb_dict.
            estimator, valid_metrics, test_metrics = self.mp._run_estimator_probe(
                self.model_name, self.data_name, self.dataset, sweep_mode=sweep_mode
            )
            return estimator, valid_metrics, test_metrics

        else:
            tokenizer = get_tokenizer(self.model_name)
            probe, valid_metrics, test_metrics = self.mp._run_nn_probe(
                self.model_name, self.data_name,
                train_set, valid_set, test_set,
                tokenizer, 
                emb_dict=self.emb_dict, 
                ppi=ppi, 
                sweep_mode=sweep_mode
            )
            return probe, valid_metrics, test_metrics

    def select_metric(self, valid_metrics: dict[str, Any], test_metrics: dict[str, Any], sweep_metric: str) -> float:
        if valid_metrics and sweep_metric in valid_metrics:
            return float(valid_metrics[sweep_metric])
        elif test_metrics and sweep_metric in test_metrics:
            return float(test_metrics[sweep_metric])
        
        # Raise a helpful error if metric was not found
        available_keys = []
        if valid_metrics:
            available_keys.extend(valid_metrics.keys())
        if test_metrics:
            available_keys.extend(test_metrics.keys())
        raise KeyError(f"Metric '{sweep_metric}' not found in validation or test metrics. Available metrics: {available_keys}")

    def objective(self, trial: Any) -> float:
        """One trial: draw a configuration, train on it, score it, and add it to the sweep's trial table."""
        full_config = {name: suggest(trial, name, spec) for name, spec in self.parameters.items()}
        self.apply_config(full_config)
        # Filter to only include the hyperparameters that were actually tuned
        applied_config = {k: v for k, v in full_config.items() if k in self.swept_param_keys}
        self.mp.trainer_args.make_plots = False

        embedding_config_changed = (
            'embedding_pooling_types' in full_config
            or 'embedding_hidden_state_index' in full_config
        )
        if embedding_config_changed and not self.mp.full_args.full_finetuning:
            _, _, _, _, _, ppi = self.dataset
            self.reload_embeddings(ppi)

        # A trial's curve would restart the sweep run's steps; its row in the trial table is its record.
        with curves_closed():
            _, valid_metrics, test_metrics = self.train_model(sweep_mode=True)

        # Choose task-specific metric to optimize
        label_type = self.mp.probe_args.task_type
        metric_cls = getattr(self.mp.full_args, 'sweep_metric_cls', None)
        metric_reg = getattr(self.mp.full_args, 'sweep_metric_reg', None)
        dataset_metric = metric_cls if label_type in ["singlelabel", "multilabel"] else metric_reg
        metric_value = self.select_metric(valid_metrics, test_metrics, dataset_metric)

        self.results_list.append({
            "trial": trial.number,
            dataset_metric: metric_value,
            "config": applied_config,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        })
        job = open_job()
        if job is not None:
            job.run.table("tuning/trials", [
                {"trial": scored["trial"], dataset_metric: scored[dataset_metric], **scored["config"]} for scored in self.results_list
            ])
        return float(metric_value)

    @classmethod
    def run_wandb_hyperopt(cls, mp: MainProcess) -> None:
        mp.logger.info("Called method: run_wandb_hyperopt")

        sweep_config = {}
        sweep_config_path = mp.full_args.sweep_config_path
            
        if os.path.exists(sweep_config_path):
            with open(sweep_config_path) as f:
                sweep_config = yaml.safe_load(f)
        else:
            raise ValueError(f"Sweep config file not found: {sweep_config_path}")

        params_to_hyperopt = sweep_config.get("parameters", {})
        
        # Filter parameters based on probe type and LoRA settings
        probe_type = getattr(mp.probe_args, 'probe_type', 'mlp')
        tokenwise = getattr(mp.probe_args, 'tokenwise', False)
        use_lora = getattr(mp.probe_args, 'lora', False)
        
        # Define which parameters are relevant for each probe type
        shared_params = {'probe_type', 'embedding_pooling_types', 'embedding_hidden_state_index', 'random_pair_flipping'}
        mlp_probe_params = {'lr', 'weight_decay', 'hidden_size', 'n_layers', 'dropout', 'pre_ln', 'use_bias', 'probe_batch_size'} | shared_params
        transformer_probe_params = {'lr', 'weight_decay', 'hidden_size', 'n_layers', 'transformer_dropout', 'pre_ln',
                                     'classifier_dropout', 'classifier_size', 'use_bias', 'probe_pooling_types', 'probe_batch_size',
                                     'bom_k', 'head_size', 'probe_grad_accum', 'add_token_ids'} | shared_params
        estimator_probe_params = {'max_depth', 'n_estimators', 'learning_rate', 'min_child_weight', 'colsample_bytree',
                                   'subsample', 'reg_lambda', 'reg_alpha', 'gamma', 'num_leaves', 'max_features',
                                   'min_samples_leaf'} | shared_params
        lora_params = {'lora_r', 'lora_alpha', 'lora_dropout'}

        # Determine which parameters to include. Sweeping probe_type is the point of a
        # neural-against-estimator comparison, and a trial can then land on either family,
        # so the allowlist has to admit both. A parameter that does not apply to the family
        # a given trial picked is simply never read.
        if 'probe_type' in params_to_hyperopt:
            relevant_params = mlp_probe_params | transformer_probe_params | estimator_probe_params
        elif probe_type == 'mlp':
            relevant_params = mlp_probe_params
        elif probe_type == 'transformer':
            relevant_params = transformer_probe_params
        elif is_estimator_probe(probe_type):
            relevant_params = estimator_probe_params
        else:
            relevant_params = mlp_probe_params | transformer_probe_params

        # Add LoRA parameters only if LoRA is enabled
        if use_lora:
            relevant_params = relevant_params | lora_params
        
        # Filter the parameters dictionary
        filtered_params = {k: v for k, v in params_to_hyperopt.items() if k in relevant_params}
        params_to_hyperopt = filtered_params

        # A trial lands on the configured probe type, or on any one the sweep lists.
        trial_probe_types = [probe_type]
        if 'probe_type' in params_to_hyperopt:
            probe_type_spec = params_to_hyperopt['probe_type']
            trial_probe_types = [
                normalize_probe_type(name) for name in probe_type_spec.get('values') or [probe_type_spec['value']]
            ]

        # Offer only values that every probe type a trial can land on takes, so no trial stops
        # before it trains. Lyra takes one pooling type, only the transformer sequence probe takes
        # bom, and the transformer probe splits hidden_size into heads of head_size. A parameter
        # left with no value is not swept, so the run's configured value stays.
        narrowed = {}
        if 'probe_pooling_types' in params_to_hyperopt:
            narrowed['probe_pooling_types'] = _pooling_every_probe_takes(
                params_to_hyperopt['probe_pooling_types'], trial_probe_types, tokenwise
            )
        if 'hidden_size' in params_to_hyperopt and 'transformer' in trial_probe_types:
            narrowed['hidden_size'] = _hidden_sizes_heads_divide(
                params_to_hyperopt['hidden_size'],
                params_to_hyperopt.get('head_size', {'value': mp.probe_args.head_size}),
            )
        for name, spec in narrowed.items():
            if spec is None:
                del params_to_hyperopt[name]
            else:
                params_to_hyperopt[name] = spec
            mp.logger.info(f"{name} offered to {trial_probe_types}: {spec}")

        # Log which parameters are being swept
        mp.logger.info(f"Probe type: {probe_type}, LoRA enabled: {use_lora}")
        mp.logger.info(f"Sweeping over {len(params_to_hyperopt)} parameters: {list(params_to_hyperopt.keys())}")

        method = mp.full_args.sweep_method
        early_term = sweep_config.get("early_terminate", None)

        total_combinations = len(mp.model_args.model_names) * len(mp.datasets)
        mp.logger.info(f"Hyperopt over {total_combinations} model/dataset combinations")
        for model_name in mp.model_args.model_names:
            tokenizer = get_tokenizer(model_name)
            test_seq = mp.all_seqs[0]

            if "random" in model_name.lower() or "onehot" in model_name.lower():
                print_message(f"Skipping hyperparameter optimization for {model_name}.")

                for data_name, dataset in mp.datasets.items():
                    train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                    mp.probe_args.num_labels = num_labels
                    mp.probe_args.task_type = label_type
                    mp.trainer_args.task_type = label_type
                    mp.trainer_args.make_plots = True

                    emb_dict = None
                    if not mp.full_args.full_finetuning:
                        if mp._sql:
                            input_dim = mp.get_embedding_dim_sql(mp._embedding_store(model_name), test_seq, tokenizer)
                        else:
                            emb_dict = mp._embedding_dict(model_name, mp.all_seqs)
                            input_dim = mp.get_embedding_dim_pth(emb_dict, test_seq, tokenizer)
                        mp.probe_args.input_size = input_dim * 2 if (ppi and not mp._full) else input_dim
                        mp.probe_args.max_seq_len = mp._max_length * 2 if (ppi and mp._full) else mp._max_length
                    if mp.full_args.full_finetuning:
                        _ = mp._run_full_finetuning(model_name, data_name, train_set, valid_set, test_set, ppi, sweep_mode=False)
                    elif mp.full_args.hybrid_probe:
                        _ = mp._run_hybrid_probe(model_name, data_name, train_set, valid_set, test_set, tokenizer, emb_dict=emb_dict, ppi=ppi, sweep_mode=False)
                    else:
                        _ = mp._run_nn_probe(model_name, data_name, train_set, valid_set, test_set, tokenizer, emb_dict=emb_dict, ppi=ppi, sweep_mode=False)
                continue

            for data_name, dataset in mp.datasets.items():
                mp.logger.info(f"Sweeping over {data_name} with {model_name}")
                train_set, _, _, num_labels, label_type, ppi = dataset
                mp.probe_args.num_labels = num_labels
                mp.probe_args.task_type = label_type
                mp.trainer_args.task_type = label_type

                emb_dict = None
                if not mp.full_args.full_finetuning:
                    if mp._sql:
                        input_dim = mp.get_embedding_dim_sql(mp._embedding_store(model_name), test_seq, tokenizer)
                    else:
                        emb_dict = mp._embedding_dict(model_name, mp.all_seqs)
                        input_dim = mp.get_embedding_dim_pth(emb_dict, test_seq, tokenizer)
                    mp.probe_args.input_size = input_dim * 2 if (ppi and not mp._full) else input_dim
                    mp.probe_args.max_seq_len = mp._max_length * 2 if (ppi and mp._full) else mp._max_length

                # Save base args for restoring after each trial
                base_probe = copy.deepcopy(mp.probe_args.__dict__)
                base_trainer = copy.deepcopy(mp.trainer_args.__dict__)

                results_list = []
                # Choose task-specific metric to optimize
                metric_cls = getattr(mp.full_args, 'sweep_metric_cls', None)
                metric_reg = getattr(mp.full_args, 'sweep_metric_reg', None)
                dataset_metric = metric_cls if label_type in ["singlelabel", "multilabel"] else metric_reg
                
                hyperopt_module = cls(
                    main_process=mp,
                    model_name=model_name,
                    data_name=data_name,
                    dataset=dataset,
                    emb_dict=emb_dict,
                    sweep_config=sweep_config,
                    results_list=results_list,
                    swept_param_keys=list(params_to_hyperopt.keys()),
                    parameters=params_to_hyperopt,
                )

                wb_sweep = {
                    "method": method,
                    "metric": {"name": dataset_metric, "goal": mp.full_args.sweep_goal},
                    "early_terminate": early_term,
                    "parameters": params_to_hyperopt,
                }
                cli_name = getattr(mp.full_args, 'sweep_name', None)
                if cli_name:
                    wb_sweep["name"] = cli_name
                elif "name" in sweep_config:
                    wb_sweep["name"] = sweep_config["name"]
                study = sweep_search(wb_sweep, seed=mp.trainer_args.seed)
                if early_term:
                    mp.logger.info("early_terminate is not applied: a trial reports its metrics once, when it ends.")
                # One W&B run per sweep (docs/conventions/wandb.md): trials are rows of its trial table, and
                # the final fit with the best configuration charts its curve.
                run_name = f"{wb_sweep.get('name') or 'sweep'}-{data_name}-{model_name}"
                with training_job(mp.trainer_args, model_name, data_name, {"sweep": wb_sweep}, name=run_name, force=True) as job:
                    study.optimize(hyperopt_module.objective, n_trials=mp.full_args.sweep_count)

                    # Sort, write, and save sweep results
                    reverse_flag = mp.full_args.sweep_goal == "maximize"
                    results_list.sort(key=lambda x: x[dataset_metric], reverse=reverse_flag)
                    sweep_log_path = os.path.join(mp.full_args.log_dir, f"{mp.random_id}_sweep_{data_name}_{model_name}.csv")
                    with open(sweep_log_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f, delimiter=',')
                        # Columns
                        columns = ["rank","trial",dataset_metric,"config","valid_metrics","test_metrics"]
                        writer.writerow(columns)
                        for idx, res in enumerate(results_list, start=1):
                            writer.writerow([
                                idx,
                                res['trial'],
                                res[dataset_metric],
                                json.dumps(res['config']),
                                json.dumps(res['valid_metrics']),
                                json.dumps(res['test_metrics']),
                            ])

                    # Log best hyperparameters
                    best = results_list[0] if results_list else None
                    best_score = best[dataset_metric]
                    best_config = best['config']
                    print_message(f"Best sweep result - {dataset_metric}: {best_score}")
                    print_message(f"Best hyperparameters: {json.dumps(best_config, indent=2)}")

                    # Restore base args then apply best
                    mp.probe_args.__dict__.update(copy.deepcopy(base_probe))
                    mp.trainer_args.__dict__.update(copy.deepcopy(base_trainer))
                    hyperopt_module.apply_config(best_config)
                    mp.trainer_args.make_plots = True
                    best_embedding_config_changed = (
                        'embedding_pooling_types' in best_config
                        or 'embedding_hidden_state_index' in best_config
                    )
                    if best_embedding_config_changed and not mp.full_args.full_finetuning:
                        hyperopt_module.reload_embeddings(ppi)
                
                    final_config = {
                        **best_config,
                        'probe_batch_size': mp.trainer_args.probe_batch_size,
                        'seed': mp.trainer_args.seed,
                        'patience': mp.trainer_args.patience,
                        'num_epochs': mp.trainer_args.num_epochs,
                    }
                    print_message(f"Final training config: {json.dumps(final_config, indent=2)}")

                    # Run best model with the best hyperparameters, log metrics, create plots
                    _, valid_metrics, test_metrics = hyperopt_module.train_model(sweep_mode=False)
                    job.run.finish(summary=job_summary(valid_metrics, test_metrics))
