import logging
import functools
import json
import csv
import os
import datetime
import ast
import random
import string
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from types import SimpleNamespace

try:
    from utils import print_message
except ImportError:
    from .utils import print_message


PROBE_RESULT_IDENTITY_KEYS = (
    "probe_type",
    "hidden_size",
    "dropout",
    "n_layers",
    "task_type",
    "num_labels",
)


def log_method_calls(func: Callable) -> Callable:
    """Decorator to log each call of the decorated method."""
    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        self.logger.info(f"Called method: {func.__name__}")
        return func(self, *args, **kwargs)
    return wrapper


class MetricsLogger:
    """
    Logs method calls to a text file, and keeps a TSV-based matrix of metrics:
      - Rows = dataset names
      - Columns = model names
      - Cells = JSON-encoded dictionaries of metrics
    """

    def __init__(self, args: Any) -> None:
        self.logger_args = args
        self._section_break = '\n' + '=' * 55 + '\n'

    def _start_file(self) -> None:
        args = self.logger_args
        self.log_dir = args.log_dir
        self.results_dir = args.results_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Check if PROTIFY_JOB_ID is set (from Modal app), use it if available
        protify_job_id = os.environ.get("PROTIFY_JOB_ID")
        if protify_job_id:
            self.random_id = protify_job_id
        elif args.replay_path is not None:
            self.random_id = 'replay_' + args.replay_path.split('/')[-1].split('.')[0]
        else:
            # Generate random ID with date and 4-letter code
            random_letters = ''.join(random.choices(string.ascii_uppercase, k=4))
            date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
            self.random_id = f"{date_str}_{random_letters}"
        
        self.log_file = os.path.join(self.log_dir, f"{self.random_id}.txt")
        self.results_file = os.path.join(self.results_dir, f"{self.random_id}.tsv")

    def _minimial_logger(self) -> None:
        # Set up a minimal logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Avoid adding multiple handlers if re-instantiated
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file, mode='a')
            handler.setLevel(logging.INFO)
            # Simple formatter without duplicating date/time
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # TSV tracking
        self.results_file = self.results_file
        self.logger_data_tracking = {}  # { dataset_name: { model_name: metrics_dict } }

    def _write_args(self) -> None:
        with open(self.log_file, 'a') as f:
            f.write(self._section_break)
            for k, v in self.logger_args.__dict__.items():
                if 'token' not in k.lower() and 'api' not in k.lower():
                    f.write(f"{k}:\t{v}\n")
            f.write(self._section_break)

    def start_log_main(self) -> None:
        self._start_file()

        with open(self.log_file, 'w') as f:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if self.logger_args.replay_path is not None:
                message = f'=== REPLAY OF {self.logger_args.replay_path} ===\n'
                f.write(message)
            header = f"=== Logging session started at {now} ===\n"
            f.write(header)
            self._write_args()

        self._minimial_logger()

    def start_log_gui(self) -> None:
        self._start_file()
        with open(self.log_file, 'w') as f:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if self.logger_args.replay_path is not None:
                message = f'=== REPLAY OF {self.logger_args.replay_path} ===\n'
                f.write(message)
            header = f"=== Logging session started at {now} ===\n"
            f.write(header)
            f.write(self._section_break)
        self._minimial_logger()

    def load_tsv(self) -> None:
        """Load existing TSV data into self.logger_data_tracking (row=dataset, col=model)."""
        with open(self.results_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            if not header:
                return

            model_names = header[1:]
            for row in reader:
                if row:
                    ds = row[0]
                    self.logger_data_tracking[ds] = {}
                    for i, model in enumerate(model_names, start=1):
                        cell_val = row[i].strip()
                        if cell_val:
                            try:
                                self.logger_data_tracking[ds][model] = json.loads(cell_val)
                            except json.JSONDecodeError:
                                self.logger_data_tracking[ds][model] = {"_raw": cell_val}

    def write_results(self) -> None:
        # Get all unique datasets and models
        datasets = sorted(self.logger_data_tracking.keys())
        all_models = set()
        for ds_data in self.logger_data_tracking.values():
            all_models.update(ds_data.keys())
        
        # Calculate average eval_loss for each model
        model_scores = {}
        for model in all_models:
            losses = []
            for ds in datasets:
                if ds in self.logger_data_tracking and model in self.logger_data_tracking[ds]:
                    metrics = self.logger_data_tracking[ds][model]
                    
                    # Try to get eval_loss, handling both numeric and string formats
                    eval_loss = None
                    if 'eval_loss_mean' in metrics:
                        eval_loss = metrics['eval_loss_mean']
                    elif 'eval_loss' in metrics:
                        loss_val = metrics['eval_loss']
                        # Check if it's a string
                        if isinstance(loss_val, str):
                            # Parse the mean
                            try:
                                eval_loss = float(loss_val.split('±')[0])
                            except (ValueError, IndexError):
                                continue
                        else:
                            eval_loss = loss_val
                    
                    if eval_loss is not None:
                        losses.append(eval_loss)
            
            if losses:
                model_scores[model] = sum(losses) / len(losses)
            else:
                model_scores[model] = float('inf')  # Models without eval_loss go last
        
        # Sort models by average eval_loss
        model_names = sorted(model_scores.keys(), key=lambda m: model_scores[m])

        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["dataset"] + model_names)
            for ds in datasets:
                row = [ds]
                for model in model_names:
                    # Get metrics if they exist, otherwise empty dict
                    metrics = self.logger_data_tracking.get(ds, {}).get(model, {})
                    row.append(json.dumps(metrics))
                writer.writerow(row)

    def _with_probe_result_identity(self, metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        enriched_metrics = dict(metrics_dict)
        logger_args_dict = self.logger_args.__dict__
        for key in PROBE_RESULT_IDENTITY_KEYS:
            if key in enriched_metrics:
                continue
            if key in logger_args_dict:
                enriched_metrics[key] = logger_args_dict[key]
        return enriched_metrics

    def log_metrics(self, dataset: str, model: str, metrics_dict: Dict[str, Any], split_name: Optional[str] = None) -> None:
        try:
            training_time = metrics_dict.get('training_time_seconds')
            preserve_keys = {
                'training_time_seconds',
                'training_time_seconds_mean',
                'training_time_seconds_std',
                'parallel_probe_seconds_per_run',
                'parallel_probe_group_runtime_records',
            }
            # Filter out other time-related keys, but preserve training_time_seconds variants
            filtered_dict = {k: v for k, v in metrics_dict.items() 
                           if not (('time' in k.lower() and k not in preserve_keys) or 
                                  ('second' in k.lower() and k not in preserve_keys))}
            if training_time is not None:
                # Add it in the end
                filtered_dict.pop('training_time_seconds', None)
                filtered_dict['training_time_seconds'] = training_time

            metrics_dict = self._with_probe_result_identity(filtered_dict)

            # Log the metrics
            if split_name is not None:
                self.logger.info(f"Storing metrics for {dataset}/{model} ({split_name}): {metrics_dict}")
            else:
                self.logger.info(f"Storing metrics for {dataset}/{model}: {metrics_dict}")
            
            # Initialize nested dictionaries if they don't exist
            if dataset not in self.logger_data_tracking:
                self.logger_data_tracking[dataset] = {}
            
            # Store the metrics
            self.logger_data_tracking[dataset][model] = metrics_dict
            
            # Write results after each update to ensure nothing is lost
            self.write_results()
            
        except Exception as e:
            self.logger.error(f"Error logging metrics for {dataset}/{model}: {str(e)}")

    def end_log(self) -> None:
        # Try multiple commands to get pip list
        pip_commands = [
            'python -m pip list',
            'py -m pip list',
            'pip list',
            'pip3 list',
            f'{sys.executable} -m pip list'  # Use current Python interpreter
        ]
        
        pip_list = "Could not retrieve pip list"
        for cmd in pip_commands:
            try:
                process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if process.returncode == 0 and process.stdout.strip():
                    pip_list = process.stdout.strip()
                    break
            except Exception:
                continue

        # Try to get nvidia-smi output, handle case where it's not available
        try:
            nvidia_info = os.popen('nvidia-smi').read().strip()
        except:
            nvidia_info = "nvidia-smi not available"
        
        # Get system info
        import platform
        system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine()
        }
        
        # Get Python version and executable path
        python_version = platform.python_version()
        python_executable = sys.executable
        
        # Log all information with proper formatting
        self.logger.info(self._section_break)
        self.logger.info("System Information:")
        self.logger.info(f"Python Version: {python_version}")
        self.logger.info(f"Python Executable: {python_executable}")
        for key, value in system_info.items():
            self.logger.info(f"{key.title()}: {value}")
        
        self.logger.info("\nInstalled Packages:")
        self.logger.info(pip_list)
        
        self.logger.info("\nGPU Information:")
        self.logger.info(nvidia_info)
        self.logger.info(self._section_break)


class LogReplayer:
    def __init__(self, log_file_path: str) -> None:
        self.log_file = Path(log_file_path)
        self.arguments: Dict[str, Any] = {}
        self.method_calls: List[str] = []

    def parse_log(self) -> SimpleNamespace:
        """
        Reads the log file line by line. Extracts:
          1) Global arguments into self.arguments
          2) Method calls into self.method_calls (in order)
        """
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_file}")

        with open(self.log_file, 'r') as file:
            header = next(file)
            for line in file:
                if line.startswith('='):
                    continue
                elif line.startswith('INFO'):
                    method = line.split(': ')[-1].strip()
                    self.method_calls.append(method)
                elif ':\t' in line:
                    key, value = line.split(':\t')
                    key, value = key.strip(), value.strip()
                    try:
                        value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        pass
                    self.arguments[key] = value

        return SimpleNamespace(**self.arguments)

    def run_replay(self, target_obj: Any) -> None:
        """
        Replays the collected method calls on `target_obj`.
        `target_obj` is an instance of the class/script that we want to replay.
        """
        for method in self.method_calls:
            print_message(f"Replaying call to: {method}()")
            func = getattr(target_obj, method, None)
            if not func:
                print_message(f"Warning: {method} not found on target object.")
                continue
            func()
