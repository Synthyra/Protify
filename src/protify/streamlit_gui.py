import streamlit as st
import os

# Set environment variables before importing torch to prevent Streamlit compatibility issues
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Disable Streamlit's file watcher for torch modules to prevent the RuntimeError
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

try:
    import torch
except RuntimeError as e:
    if "no running event loop" in str(e) or "__path__._path" in str(e):
        st.error("""
        **PyTorch-Streamlit Compatibility Issue Detected**
        
        This error occurs due to a known compatibility issue between PyTorch and Streamlit's file watcher.
        
        **Solutions:**
        1. Use the launcher script: `python run_streamlit_gui.py`
        2. Or run with: `STREAMLIT_SERVER_FILE_WATCHER_TYPE=none streamlit run src/protify/streamlit_gui.py`
        3. Or restart Streamlit with the file watcher disabled
        
        The application may still work, but you might experience some issues.
        """)
        # Try to continue anyway
        import torch
    else:
        raise e


import argparse
import traceback
from types import SimpleNamespace
from base_models.get_base_models import BaseModelArguments, standard_models
from data.supported_datasets import supported_datasets, standard_data_benchmark, internal_datasets
from embedder import EmbeddingArguments
from probes.get_probe import ProbeArguments
from probes.trainers import TrainerArguments
from main import MainProcess
from data.data_mixin import DataArguments
from probes.scikit_classes import ScikitArguments
from visualization.plot_result import create_plots


@st.cache_data
def get_standard_models():
    """Cache the standard models list"""
    return standard_models


@st.cache_data  
def get_available_datasets():
    """Cache the available datasets list"""
    return [d for d in supported_datasets if d not in internal_datasets]


class StreamlitGUI(MainProcess):
    def __init__(self):
        # Initialize session state only once
        if 'gui_initialized' not in st.session_state:
            # First time initialization
            super().__init__(argparse.Namespace(), GUI=True)
            st.session_state.gui_initialized = True
            st.session_state.gui_instance = self
            st.session_state.settings_vars = {}
            st.session_state.current_task = None
            st.session_state.task_running = False
            st.session_state.session_started = False
            st.session_state.models_selected = False
            st.session_state.data_loaded = False
            st.session_state.embeddings_saved = False
            st.session_state.probe_args_created = False
        else:
            # Restore from session state - just copy the attributes
            gui_instance = st.session_state.gui_instance
            if gui_instance is not None:
                # Copy all attributes from the stored instance
                for key, value in gui_instance.__dict__.items():
                    setattr(self, key, value)
                
                # Update the stored instance reference
                st.session_state.gui_instance = self

    def _ensure_logger_initialized(self):
        """Ensure logger is properly initialized before writing args"""
        if not hasattr(self, 'log_file') or not hasattr(self, 'logger'):
            # Logger not initialized, skip writing args
            return False
        return True

    def _update_session_state(self):
        """Update the session state with current instance"""
        # Only update if the instance has changed or important attributes are missing
        if (st.session_state.gui_instance is None or 
            id(st.session_state.gui_instance) != id(self) or
            not hasattr(st.session_state.gui_instance, 'random_id')):
            st.session_state.gui_instance = self

    def run_task(self, task_func, *args, **kwargs):
        """Run a task and handle errors"""
        try:
            st.session_state.task_running = True
            result = task_func(*args, **kwargs)
            st.session_state.task_running = False
            return result
        except Exception as e:
            st.session_state.task_running = False
            st.error(f"Error: {str(e)}")
            traceback.print_exc()
            return None

    def build_sidebar(self):
        """Build the sidebar navigation"""
        with st.sidebar:
            # Only load image once
            if 'logo_loaded' not in st.session_state:
                try:
                    st.image("synthyra_logo.png", width=150)
                    st.session_state.logo_loaded = True
                except:
                    pass  # Logo file not found, continue without it
            elif st.session_state.logo_loaded:
                try:
                    st.image("synthyra_logo.png", width=150)
                except:
                    pass
            
            st.title("Navigation")
            
            page = st.radio(
                "Select Page",
                ["Info", "Model", "Data", "Embedding", "Probe", "Trainer", "Scikit", "Replay", "Visualization"],
                key="page_selection"
            )
            
            st.markdown("---")
            st.markdown("[Visit Synthyra.com](https://synthyra.com)")
            
        return page

    def build_info_page(self):
        st.header("Protify Session Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Identification")
            
            hf_username = st.text_input(
                "Huggingface Username",
                value=st.session_state.settings_vars.get("huggingface_username", "Synthyra"),
                help="Your Hugging Face username for model downloads and uploads.",
                key="huggingface_username"
            )
            
            hf_token = st.text_input(
                "Huggingface Token",
                value=st.session_state.settings_vars.get("huggingface_token", ""),
                type="password",
                help="Your Hugging Face API token for accessing gated or private models.",
                key="huggingface_token"
            )
            
            wandb_api_key = st.text_input(
                "Wandb API Key",
                value=st.session_state.settings_vars.get("wandb_api_key", ""),
                type="password",
                help="Your Weights & Biases API key for experiment tracking.",
                key="wandb_api_key"
            )
            
            synthyra_api_key = st.text_input(
                "Synthyra API Key",
                value=st.session_state.settings_vars.get("synthyra_api_key", ""),
                type="password",
                help="Your Synthyra API key for accessing premium features.",
                key="synthyra_api_key"
            )
        
        with col2:
            st.subheader("Paths")
            
            home_dir = st.text_input(
                "Home Directory",
                value=st.session_state.settings_vars.get("home_dir", os.getcwd()),
                help="Home directory for Protify.",
                key="home_dir"
            )
            
            log_dir = st.text_input(
                "Log Directory",
                value=st.session_state.settings_vars.get("log_dir", "logs"),
                help="Directory where log files will be stored.",
                key="log_dir"
            )
            
            results_dir = st.text_input(
                "Results Directory",
                value=st.session_state.settings_vars.get("results_dir", "results"),
                help="Directory where results data will be stored.",
                key="results_dir"
            )
            
            model_save_dir = st.text_input(
                "Model Save Directory",
                value=st.session_state.settings_vars.get("model_save_dir", "weights"),
                help="Directory where trained models will be saved.",
                key="model_save_dir"
            )
            
            plots_dir = st.text_input(
                "Plots Directory",
                value=st.session_state.settings_vars.get("plots_dir", "plots"),
                help="Directory where plots and visualizations will be saved.",
                key="plots_dir"
            )
            
            embedding_save_dir = st.text_input(
                "Embedding Save Directory",
                value=st.session_state.settings_vars.get("embedding_save_dir", "embeddings"),
                help="Directory where computed embeddings will be saved.",
                key="embedding_save_dir"
            )
            
            download_dir = st.text_input(
                "Download Directory",
                value=st.session_state.settings_vars.get("download_dir", "Synthyra/mean_pooled_embeddings"),
                help="HuggingFace repository path for downloading pre-computed embeddings.",
                key="download_dir"
            )
        
        # Update settings vars
        st.session_state.settings_vars.update({
            "huggingface_username": hf_username,
            "huggingface_token": hf_token,
            "wandb_api_key": wandb_api_key,
            "synthyra_api_key": synthyra_api_key,
            "home_dir": home_dir,
            "log_dir": log_dir,
            "results_dir": results_dir,
            "model_save_dir": model_save_dir,
            "plots_dir": plots_dir,
            "embedding_save_dir": embedding_save_dir,
            "download_dir": download_dir
        })
        
        if st.button("Start Session", type="primary", disabled=st.session_state.task_running):
            with st.spinner("Starting session..."):
                self._session_start()

    def build_model_page(self):
        st.header("Model Selection")
        
        if not st.session_state.session_started:
            st.warning("Please start a session first on the Info page.")
            return
        
        st.write("Select the language models to use for embedding. Multiple models can be selected.")
        
        selected_models = st.multiselect(
            "Available Models",
            options=get_standard_models(),
            default=st.session_state.settings_vars.get("model_names", []),
            help="Select the language models to use for embedding. Multiple models can be selected.",
            key="model_selection"
        )
        
        st.session_state.settings_vars["model_names"] = selected_models
        
        if st.button("Select Models", type="primary", disabled=st.session_state.task_running):
            if not selected_models:
                st.warning("No models selected. Using all standard models.")
            with st.spinner("Configuring models..."):
                self._select_models()

    def build_data_page(self):
        st.header("Data Configuration")
        
        if not st.session_state.session_started:
            st.warning("Please start a session first on the Info page.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_length = st.number_input(
                "Max Sequence Length",
                min_value=1,
                max_value=32768,
                value=st.session_state.settings_vars.get("max_length", 1024),
                help="Maximum length of sequences (in tokens) to process.",
                key="max_length"
            )
            
            trim = st.checkbox(
                "Trim Sequences",
                value=st.session_state.settings_vars.get("trim", False),
                help="Whether to trim sequences to the specified max length.",
                key="trim"
            )
            
            delimiter = st.text_input(
                "Delimiter",
                value=st.session_state.settings_vars.get("delimiter", ","),
                help="Character used to separate columns in CSV data files.",
                key="delimiter"
            )
            
            col_names = st.text_input(
                "Column Names (comma-separated)",
                value=st.session_state.settings_vars.get("col_names", "seqs,labels"),
                help="Names of columns in data files, separate with commas.",
                key="col_names"
            )
        
        with col2:
            st.write("Select datasets to use. Multiple datasets can be selected.")
            
            available_datasets = get_available_datasets()
            selected_datasets = st.multiselect(
                "Available Datasets",
                options=available_datasets,
                default=st.session_state.settings_vars.get("data_names", []),
                help="Select datasets to use. Multiple datasets can be selected.",
                key="dataset_selection"
            )
        
        # Update settings
        st.session_state.settings_vars.update({
            "max_length": max_length,
            "trim": trim,
            "delimiter": delimiter,
            "col_names": col_names,
            "data_names": selected_datasets
        })
        
        if st.button("Get Data", type="primary", disabled=st.session_state.task_running):
            if not selected_datasets:
                st.warning("No datasets selected. Using standard data benchmark.")
            with st.spinner("Loading datasets..."):
                self._get_data()

    def build_embedding_page(self):
        st.header("Embedding Configuration")
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first using the Data tab.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=1024,
                value=st.session_state.settings_vars.get("batch_size", 4),
                help="Number of sequences to process at once during embedding.",
                key="batch_size"
            )
            
            num_workers = st.number_input(
                "Num Workers",
                min_value=0,
                max_value=64,
                value=st.session_state.settings_vars.get("num_workers", 0),
                help="Number of worker processes for data loading. 0 means main process only.",
                key="num_workers"
            )
            
            download_embeddings = st.checkbox(
                "Download Embeddings",
                value=st.session_state.settings_vars.get("download_embeddings", False),
                help="Whether to download pre-computed embeddings from HuggingFace instead of computing them.",
                key="download_embeddings"
            )
            
            matrix_embed = st.checkbox(
                "Matrix Embedding",
                value=st.session_state.settings_vars.get("matrix_embed", False),
                help="Whether to use matrix embedding (full embedding matrices) instead of pooled embeddings.",
                key="matrix_embed"
            )
        
        with col2:
            pooling_types = st.text_input(
                "Pooling Types (comma-separated)",
                value=st.session_state.settings_vars.get("embedding_pooling_types", "mean"),
                help="Types of pooling to apply to embeddings, separate with commas.",
                key="embedding_pooling_types"
            )
            
            st.caption("Options: mean, max, min, norm, prod, median, std, var, cls, parti")
            
            embed_dtype = st.selectbox(
                "Embedding DType",
                options=["float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"],
                index=0,
                help="Data type to use for storing embeddings (affects precision and size).",
                key="embed_dtype"
            )
            
            sql = st.checkbox(
                "Use SQL",
                value=st.session_state.settings_vars.get("sql", False),
                help="Whether to use SQL database for storing embeddings instead of files.",
                key="sql"
            )
        
        # Update settings
        st.session_state.settings_vars.update({
            "batch_size": batch_size,
            "num_workers": num_workers,
            "download_embeddings": download_embeddings,
            "matrix_embed": matrix_embed,
            "embedding_pooling_types": pooling_types,
            "embed_dtype": embed_dtype,
            "sql": sql
        })
        
        if st.button("Embed sequences to disk", type="primary", disabled=st.session_state.task_running):
            with st.spinner("Computing embeddings..."):
                self._get_embeddings()

    def build_probe_page(self):
        st.header("Probe Configuration")
        
        # Main probe settings
        col1, col2 = st.columns(2)
        
        with col1:
            probe_type = st.selectbox(
                "Probe Type",
                options=["linear", "transformer", "retrievalnet", "lyra"],
                index=0,
                help="Type of probe architecture to use (linear, transformer, or retrievalnet).",
                key="probe_type"
            )
            
            tokenwise = st.checkbox(
                "Tokenwise",
                value=st.session_state.settings_vars.get("tokenwise", False),
                help="Whether to use token-wise prediction (operate on each token) instead of sequence-level.",
                key="tokenwise"
            )
            
            pre_ln = st.checkbox(
                "Pre Layer Norm",
                value=st.session_state.settings_vars.get("pre_ln", True),
                help="Whether to use pre-layer normalization in transformer architecture.",
                key="pre_ln"
            )
            
            n_layers = st.number_input(
                "Number of Layers",
                min_value=1,
                max_value=100,
                value=st.session_state.settings_vars.get("n_layers", 1),
                help="Number of layers in the probe architecture.",
                key="n_layers"
            )
            
            hidden_dim = st.number_input(
                "Hidden Dimension",
                min_value=1,
                max_value=10000,
                value=st.session_state.settings_vars.get("hidden_dim", 8192),
                help="Size of hidden dimension in the probe model.",
                key="hidden_dim"
            )
            
            dropout = st.slider(
                "Dropout",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.settings_vars.get("dropout", 0.2),
                step=0.1,
                help="Dropout probability for regularization (0.0-1.0).",
                key="dropout"
            )
        
        with col2:
            st.subheader("Transformer Probe Settings")
            
            classifier_dim = st.number_input(
                "Classifier Dimension",
                min_value=1,
                max_value=10000,
                value=st.session_state.settings_vars.get("classifier_dim", 4096),
                help="Dimension of the classifier/feedforward layer in transformer probe.",
                key="classifier_dim"
            )
            
            classifier_dropout = st.slider(
                "Classifier Dropout",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.settings_vars.get("classifier_dropout", 0.2),
                step=0.1,
                help="Dropout probability in the classifier layer (0.0-1.0).",
                key="classifier_dropout"
            )
            
            n_heads = st.number_input(
                "Number of Heads",
                min_value=1,
                max_value=32,
                value=st.session_state.settings_vars.get("n_heads", 4),
                help="Number of attention heads in transformer probe.",
                key="n_heads"
            )
            
            rotary = st.checkbox(
                "Rotary",
                value=st.session_state.settings_vars.get("rotary", True),
                help="Whether to use rotary position embeddings in transformer.",
                key="rotary"
            )
            
            probe_pooling_types = st.text_input(
                "Pooling Types (comma-separated)",
                value=st.session_state.settings_vars.get("probe_pooling_types", "mean, cls"),
                help="Types of pooling to use in the probe model, separate with commas.",
                key="probe_pooling_types"
            )
            
            transformer_dropout = st.slider(
                "Transformer Dropout",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.settings_vars.get("transformer_dropout", 0.1),
                step=0.1,
                help="Dropout probability in the transformer layers (0.0-1.0).",
                key="transformer_dropout"
            )
        
        # Additional settings
        st.subheader("Additional Settings")
        
        col3, col4 = st.columns(2)
        
        with col3:
            save_model = st.checkbox(
                "Save Model",
                value=st.session_state.settings_vars.get("save_model", False),
                help="Whether to save the trained probe model to disk.",
                key="save_model"
            )
            
            production_model = st.checkbox(
                "Production Model",
                value=st.session_state.settings_vars.get("production_model", False),
                help="Whether to prepare the model for production deployment.",
                key="production_model"
            )
        
        with col4:
            st.subheader("LoRA Settings")
            
            use_lora = st.checkbox(
                "Use LoRA",
                value=st.session_state.settings_vars.get("use_lora", False),
                help="Whether to use Low-Rank Adaptation (LoRA) for fine-tuning.",
                key="use_lora"
            )
            
            if use_lora:
                lora_r = st.number_input(
                    "LoRA r",
                    min_value=1,
                    max_value=128,
                    value=st.session_state.settings_vars.get("lora_r", 8),
                    help="Rank parameter r for LoRA (lower = more efficient, higher = more expressive).",
                    key="lora_r"
                )
                
                lora_alpha = st.number_input(
                    "LoRA alpha",
                    min_value=1.0,
                    max_value=128.0,
                    value=st.session_state.settings_vars.get("lora_alpha", 32.0),
                    step=1.0,
                    help="Alpha parameter for LoRA, controls update scale.",
                    key="lora_alpha"
                )
                
                lora_dropout = st.slider(
                    "LoRA dropout",
                    min_value=0.0,
                    max_value=0.5,
                    value=st.session_state.settings_vars.get("lora_dropout", 0.01),
                    step=0.01,
                    help="Dropout probability for LoRA layers (0.0-0.5).",
                    key="lora_dropout"
                )
            else:
                lora_r = 8
                lora_alpha = 32.0
                lora_dropout = 0.01
        
        # Update all settings
        st.session_state.settings_vars.update({
            "probe_type": probe_type,
            "tokenwise": tokenwise,
            "pre_ln": pre_ln,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "classifier_dim": classifier_dim,
            "classifier_dropout": classifier_dropout,
            "n_heads": n_heads,
            "rotary": rotary,
            "probe_pooling_types": probe_pooling_types,
            "transformer_dropout": transformer_dropout,
            "save_model": save_model,
            "production_model": production_model,
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout
        })
        
        if st.button("Save Probe Arguments", type="primary", disabled=st.session_state.task_running):
            with st.spinner("Creating probe arguments..."):
                self._create_probe_args()

    def build_trainer_page(self):
        st.header("Trainer Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hybrid_probe = st.checkbox(
                "Hybrid Probe",
                value=st.session_state.settings_vars.get("hybrid_probe", False),
                help="Whether to use hybrid probe (combines neural and linear probes).",
                key="hybrid_probe"
            )
            
            full_finetuning = st.checkbox(
                "Full Finetuning",
                value=st.session_state.settings_vars.get("full_finetuning", False),
                help="Whether to perform full finetuning of the entire model.",
                key="full_finetuning"
            )
            
            num_epochs = st.number_input(
                "Number of Epochs",
                min_value=1,
                max_value=1000,
                value=st.session_state.settings_vars.get("num_epochs", 200),
                help="Number of training epochs (complete passes through the dataset).",
                key="num_epochs"
            )
            
            probe_batch_size = st.number_input(
                "Probe Batch Size",
                min_value=1,
                max_value=1000,
                value=st.session_state.settings_vars.get("probe_batch_size", 64),
                help="Batch size for probe training.",
                key="probe_batch_size"
            )
            
            base_batch_size = st.number_input(
                "Base Batch Size",
                min_value=1,
                max_value=1000,
                value=st.session_state.settings_vars.get("base_batch_size", 4),
                help="Batch size for base model training.",
                key="base_batch_size"
            )
        
        with col2:
            probe_grad_accum = st.number_input(
                "Probe Grad Accum",
                min_value=1,
                max_value=100,
                value=st.session_state.settings_vars.get("probe_grad_accum", 1),
                help="Gradient accumulation steps for probe training.",
                key="probe_grad_accum"
            )
            
            base_grad_accum = st.number_input(
                "Base Grad Accum",
                min_value=1,
                max_value=100,
                value=st.session_state.settings_vars.get("base_grad_accum", 8),
                help="Gradient accumulation steps for base model training.",
                key="base_grad_accum"
            )
            
            lr = st.number_input(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-2,
                value=st.session_state.settings_vars.get("lr", 1e-4),
                format="%.6f",
                help="Learning rate for optimizer. Controls step size during training.",
                key="lr"
            )
            
            weight_decay = st.slider(
                "Weight Decay",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.settings_vars.get("weight_decay", 0.0),
                step=0.01,
                help="L2 regularization factor to prevent overfitting (0.0-1.0).",
                key="weight_decay"
            )
            
            patience = st.number_input(
                "Patience",
                min_value=1,
                max_value=100,
                value=st.session_state.settings_vars.get("patience", 1),
                help="Number of epochs with no improvement after which training will stop.",
                key="patience"
            )
            
            seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=10000,
                value=st.session_state.settings_vars.get("seed", 42),
                help="Random seed for reproducibility of experiments.",
                key="seed"
            )
        
        # Update settings
        st.session_state.settings_vars.update({
            "hybrid_probe": hybrid_probe,
            "full_finetuning": full_finetuning,
            "num_epochs": num_epochs,
            "probe_batch_size": probe_batch_size,
            "base_batch_size": base_batch_size,
            "probe_grad_accum": probe_grad_accum,
            "base_grad_accum": base_grad_accum,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "seed": seed
        })
        
        if st.button("Run Trainer", type="primary", disabled=st.session_state.task_running):
            with st.spinner("Running trainer..."):
                self._run_trainer()

    def build_scikit_page(self):
        st.header("Scikit-Learn Configuration")
        
        use_scikit = st.checkbox(
            "Use Scikit",
            value=st.session_state.settings_vars.get("use_scikit", False),
            help="Whether to use scikit-learn models instead of neural networks.",
            key="use_scikit"
        )
        
        if use_scikit:
            col1, col2 = st.columns(2)
            
            with col1:
                scikit_n_iter = st.number_input(
                    "Scikit Iterations",
                    min_value=1,
                    max_value=1000,
                    value=st.session_state.settings_vars.get("scikit_n_iter", 10),
                    help="Number of iterations for iterative scikit-learn models.",
                    key="scikit_n_iter"
                )
                
                scikit_cv = st.number_input(
                    "Scikit CV Folds",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.settings_vars.get("scikit_cv", 3),
                    help="Number of cross-validation folds for model evaluation.",
                    key="scikit_cv"
                )
                
                scikit_random_state = st.number_input(
                    "Scikit Random State",
                    min_value=0,
                    max_value=10000,
                    value=st.session_state.settings_vars.get("scikit_random_state", 42),
                    help="Random seed for scikit-learn models to ensure reproducibility.",
                    key="scikit_random_state"
                )
            
            with col2:
                scikit_model_name = st.text_input(
                    "Scikit Model Name (optional)",
                    value=st.session_state.settings_vars.get("scikit_model_name", ""),
                    help="Optional name for the scikit-learn model. Leave blank to use default.",
                    key="scikit_model_name"
                )
                
                n_jobs = st.number_input(
                    "Number of Jobs",
                    min_value=-1,
                    max_value=32,
                    value=st.session_state.settings_vars.get("n_jobs", 1),
                    help="Number of CPU cores to use for parallel processing. Use -1 for all cores.",
                    key="n_jobs"
                )
        
        # Update settings
        st.session_state.settings_vars.update({
            "use_scikit": use_scikit,
            "scikit_n_iter": scikit_n_iter if use_scikit else 10,
            "scikit_cv": scikit_cv if use_scikit else 3,
            "scikit_random_state": scikit_random_state if use_scikit else 42,
            "scikit_model_name": scikit_model_name if use_scikit else "",
            "n_jobs": n_jobs if use_scikit else 1
        })
        
        if st.button("Run Scikit Models", type="primary", disabled=st.session_state.task_running or not use_scikit):
            with st.spinner("Running scikit-learn models..."):
                self._run_scikit()

    def build_replay_page(self):
        st.header("Log Replay")
        
        st.write("Select a log file to replay a previous run.")
        
        replay_file = st.file_uploader(
            "Choose a log file",
            type=['txt'],
            help="Select a log file to replay."
        )
        
        if replay_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_replay_{replay_file.name}"
            with open(temp_path, "wb") as f:
                f.write(replay_file.getbuffer())
            
            st.session_state.settings_vars["replay_path"] = temp_path
            
            if st.button("Start Replay", type="primary", disabled=st.session_state.task_running):
                with st.spinner("Running replay..."):
                    self._start_replay()
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def build_viz_page(self):
        st.header("Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            result_id = st.text_input(
                "Result ID",
                value=st.session_state.settings_vars.get("result_id", ""),
                help="ID of the result to visualize. Will look for results/{result_id}.tsv",
                key="result_id"
            )
            
            use_current_run = st.checkbox(
                "Use Current Run",
                value=st.session_state.settings_vars.get("use_current_run", True),
                help="Use results from the current run.",
                key="use_current_run"
            )
        
        with col2:
            results_file = st.file_uploader(
                "Or upload a results file",
                type=['tsv'],
                help="Upload a TSV results file directly."
            )
            
            viz_output_dir = st.text_input(
                "Output Directory",
                value=st.session_state.settings_vars.get("viz_output_dir", "plots"),
                help="Directory where plots will be saved.",
                key="viz_output_dir"
            )
        
        # Update settings
        st.session_state.settings_vars.update({
            "result_id": result_id,
            "use_current_run": use_current_run,
            "viz_output_dir": viz_output_dir
        })
        
        if results_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_results_{results_file.name}"
            with open(temp_path, "wb") as f:
                f.write(results_file.getbuffer())
            st.session_state.settings_vars["results_file"] = temp_path
            st.session_state.settings_vars["use_current_run"] = False
        
        if st.button("Generate Plots", type="primary", disabled=st.session_state.task_running):
            with st.spinner("Generating plots..."):
                self._generate_plots()

    # Helper methods adapted from original GUI
    def _session_start(self):        
        hf_token = st.session_state.settings_vars["huggingface_token"]
        synthyra_api_key = st.session_state.settings_vars["synthyra_api_key"]
        wandb_api_key = st.session_state.settings_vars["wandb_api_key"]
        
        if hf_token:
            from huggingface_hub import login
            login(hf_token)
            st.success('Logged in to Hugging Face')
        
        self.full_args.hf_username = st.session_state.settings_vars["huggingface_username"]
        self.full_args.hf_token = hf_token
        self.full_args.synthyra_api_key = synthyra_api_key
        self.full_args.wandb_api_key = wandb_api_key
        self.full_args.home_dir = st.session_state.settings_vars["home_dir"]
        
        def _make_true_dir(path):
            true_path = os.path.join(self.full_args.home_dir, path)
            os.makedirs(true_path, exist_ok=True)
            return true_path
        
        self.full_args.log_dir = _make_true_dir(st.session_state.settings_vars["log_dir"])
        self.full_args.results_dir = _make_true_dir(st.session_state.settings_vars["results_dir"])
        self.full_args.model_save_dir = _make_true_dir(st.session_state.settings_vars["model_save_dir"])
        self.full_args.plots_dir = _make_true_dir(st.session_state.settings_vars["plots_dir"])
        self.full_args.embedding_save_dir = _make_true_dir(st.session_state.settings_vars["embedding_save_dir"])
        self.full_args.download_dir = _make_true_dir(st.session_state.settings_vars["download_dir"])
        
        self.full_args.replay_path = None
        self.logger_args = SimpleNamespace(**self.full_args.__dict__)
        self.start_log_gui()
        
        # Update session state with the initialized logger
        self._update_session_state()
        
        st.session_state.session_started = True
        st.success(f"Session and logging started for id {self.random_id}")

    def _select_models(self):
        selected_models = st.session_state.settings_vars.get("model_names", [])
        
        if not selected_models:
            selected_models = standard_models
        
        self.full_args.model_names = selected_models
        self.model_args = BaseModelArguments(**self.full_args.__dict__)
        
        args_dict = {k: v for k, v in self.full_args.__dict__.items() 
                    if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        
        # Only write args if logger is properly initialized
        if self._ensure_logger_initialized():
            self._write_args()
        
        st.session_state.models_selected = True
        st.success(f"Selected {len(selected_models)} models")

    def _get_data(self):
        selected_datasets = st.session_state.settings_vars.get("data_names", [])
        
        if not selected_datasets:
            selected_datasets = standard_data_benchmark
        
        # Update full_args with data settings
        self.full_args.data_names = selected_datasets
        self.full_args.data_dirs = []
        self.full_args.max_length = st.session_state.settings_vars["max_length"]
        self.full_args.trim = st.session_state.settings_vars["trim"]
        self.full_args.delimiter = st.session_state.settings_vars["delimiter"]
        self.full_args.col_names = st.session_state.settings_vars["col_names"].split(",")
        
        # Update mixin attributes
        self._max_length = self.full_args.max_length
        self._trim = self.full_args.trim
        self._delimiter = self.full_args.delimiter
        self._col_names = self.full_args.col_names
        
        # Create data args and get datasets
        self.data_args = DataArguments(**self.full_args.__dict__)
        args_dict = {k: v for k, v in self.full_args.__dict__.items() 
                    if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        
        # Only write args if logger is properly initialized
        if self._ensure_logger_initialized():
            self._write_args()
        self.get_datasets()
        
        st.session_state.data_loaded = True
        st.success("Data downloaded and stored")

    def _get_embeddings(self):
        if not hasattr(self, 'all_seqs') or not self.all_seqs:
            st.error('Sequences are not loaded yet. Please run the data tab first.')
            return
        
        pooling_str = st.session_state.settings_vars["embedding_pooling_types"].strip()
        pooling_list = [p.strip() for p in pooling_str.split(",") if p.strip()]
        dtype_str = st.session_state.settings_vars["embed_dtype"]
        dtype_val = self.dtype_map.get(dtype_str, torch.float32)
        
        # Update full args
        self.full_args.all_seqs = self.all_seqs
        self.full_args.embedding_batch_size = st.session_state.settings_vars["batch_size"]
        self.full_args.embedding_num_workers = st.session_state.settings_vars["num_workers"]
        self.full_args.download_embeddings = st.session_state.settings_vars["download_embeddings"]
        self.full_args.matrix_embed = st.session_state.settings_vars["matrix_embed"]
        self.full_args.embedding_pooling_types = pooling_list
        self.full_args.save_embeddings = True
        self.full_args.embed_dtype = dtype_val
        self.full_args.sql = st.session_state.settings_vars["sql"]
        self._sql = self.full_args.sql
        self._full = self.full_args.matrix_embed
        
        self.embedding_args = EmbeddingArguments(**self.full_args.__dict__)
        args_dict = {k: v for k, v in self.full_args.__dict__.items() 
                    if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        
        # Only write args if logger is properly initialized
        if self._ensure_logger_initialized():
            self._write_args()
        
        self.save_embeddings_to_disk()
        
        st.session_state.embeddings_saved = True
        st.success("Embeddings saved to disk")

    def _create_probe_args(self):        
        # Convert pooling types string to list
        probe_pooling_types = [p.strip() for p in st.session_state.settings_vars["probe_pooling_types"].split(",")]
        
        # Update full_args with probe settings
        self.full_args.probe_type = st.session_state.settings_vars["probe_type"]
        self.full_args.tokenwise = st.session_state.settings_vars["tokenwise"]
        self.full_args.hidden_dim = st.session_state.settings_vars["hidden_dim"]
        self.full_args.dropout = st.session_state.settings_vars["dropout"]
        self.full_args.n_layers = st.session_state.settings_vars["n_layers"]
        self.full_args.pre_ln = st.session_state.settings_vars["pre_ln"]
        self.full_args.classifier_dim = st.session_state.settings_vars["classifier_dim"]
        self.full_args.transformer_dropout = st.session_state.settings_vars["transformer_dropout"]
        self.full_args.classifier_dropout = st.session_state.settings_vars["classifier_dropout"]
        self.full_args.n_heads = st.session_state.settings_vars["n_heads"]
        self.full_args.rotary = st.session_state.settings_vars["rotary"]
        self.full_args.probe_pooling_types = probe_pooling_types
        self.full_args.save_model = st.session_state.settings_vars["save_model"]
        self.full_args.production_model = st.session_state.settings_vars["production_model"]
        self.full_args.use_lora = st.session_state.settings_vars["use_lora"]
        self.full_args.lora_r = st.session_state.settings_vars["lora_r"]
        self.full_args.lora_alpha = st.session_state.settings_vars["lora_alpha"]
        self.full_args.lora_dropout = st.session_state.settings_vars["lora_dropout"]
        
        # Create probe args from full args
        self.probe_args = ProbeArguments(**self.full_args.__dict__)
        
        args_dict = {k: v for k, v in self.full_args.__dict__.items() 
                    if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        
        # Only write args if logger is properly initialized
        if self._ensure_logger_initialized():
            self._write_args()
        
        st.session_state.probe_args_created = True
        st.success("Probe arguments saved")

    def _run_trainer(self):        
        # Gather settings
        self.full_args.use_lora = st.session_state.settings_vars["use_lora"]
        self.full_args.hybrid_probe = st.session_state.settings_vars["hybrid_probe"]
        self.full_args.full_finetuning = st.session_state.settings_vars["full_finetuning"]
        self.full_args.lora_r = st.session_state.settings_vars.get("lora_r", 8)
        self.full_args.lora_alpha = st.session_state.settings_vars.get("lora_alpha", 32.0)
        self.full_args.lora_dropout = st.session_state.settings_vars.get("lora_dropout", 0.01)
        self.full_args.num_epochs = st.session_state.settings_vars["num_epochs"]
        self.full_args.trainer_batch_size = st.session_state.settings_vars["probe_batch_size"]
        self.full_args.gradient_accumulation_steps = st.session_state.settings_vars["probe_grad_accum"]
        self.full_args.lr = st.session_state.settings_vars["lr"]
        self.full_args.weight_decay = st.session_state.settings_vars["weight_decay"]
        self.full_args.patience = st.session_state.settings_vars["patience"]
        self.full_args.seed = st.session_state.settings_vars["seed"]
        
        self.trainer_args = TrainerArguments(**self.full_args.__dict__)
        args_dict = {k: v for k, v in self.full_args.__dict__.items() 
                    if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        
        # Only write args if logger is properly initialized
        if self._ensure_logger_initialized():
            self._write_args()
        
        if self.full_args.full_finetuning:
            self.run_full_finetuning()
        elif self.full_args.hybrid_probe:
            self.run_hybrid_probes()
        else:
            self.run_nn_probes()
        
        st.success("Training completed!")

    def _run_scikit(self):        
        # Gather settings for scikit
        self.full_args.use_scikit = st.session_state.settings_vars["use_scikit"]
        self.full_args.scikit_n_iter = st.session_state.settings_vars["scikit_n_iter"]
        self.full_args.scikit_cv = st.session_state.settings_vars["scikit_cv"]
        self.full_args.scikit_random_state = st.session_state.settings_vars["scikit_random_state"]
        self.full_args.scikit_model_name = st.session_state.settings_vars["scikit_model_name"]
        self.full_args.n_jobs = st.session_state.settings_vars["n_jobs"]
        
        self.scikit_args = ScikitArguments(**self.full_args.__dict__)
        args_dict = {k: v for k, v in self.full_args.__dict__.items() 
                    if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        
        # Only write args if logger is properly initialized
        if self._ensure_logger_initialized():
            self._write_args()
        
        self.run_scikit_scheme()
        st.success("Scikit models completed!")

    def _start_replay(self):
        replay_path = st.session_state.settings_vars.get("replay_path", "")
        if not replay_path:
            st.error("Please select a replay log file first")
            return
                
        from logger import LogReplayer
        replayer = LogReplayer(replay_path)
        replay_args = replayer.parse_log()
        replay_args.replay_path = replay_path
        
        # Create a new MainProcess instance with replay_args
        main = MainProcess(replay_args, GUI=False)
        
        # Run the replay on this MainProcess instance
        replayer.run_replay(main)
        st.success("Replay completed!")

    def _generate_plots(self):        
        # Determine which results file to use
        results_file = None
        
        if st.session_state.settings_vars["use_current_run"] and hasattr(self, 'random_id'):
            # Use the current run's random ID
            results_file = os.path.join(st.session_state.settings_vars["results_dir"], f"{self.random_id}.tsv")
        elif st.session_state.settings_vars.get("results_file"):
            # Use explicitly selected file
            results_file = st.session_state.settings_vars["results_file"]
        elif st.session_state.settings_vars["result_id"]:
            # Use the specified result ID
            result_id = st.session_state.settings_vars["result_id"]
            results_file = os.path.join(st.session_state.settings_vars["results_dir"], f"{result_id}.tsv")
        else:
            st.error("No results file specified. Please enter a Result ID, upload a file, or complete a run first.")
            return
        
        # Check if the results file exists
        if not os.path.exists(results_file):
            st.error(f"Results file not found: {results_file}")
            return
        
        # Get output directory
        output_dir = st.session_state.settings_vars["viz_output_dir"]
        
        # Call the plot generation function
        create_plots(results_file, output_dir)
        st.success("Plots generated successfully!")
        
        # Clean up temp file if it exists
        if st.session_state.settings_vars.get("results_file", "").startswith("temp_"):
            os.remove(st.session_state.settings_vars["results_file"])

    def run(self):
        st.set_page_config(
            page_title="Protify",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Debug info (can be removed in production)
        if st.checkbox("Show debug info", value=False):
            st.info(f"GUI initialized: {st.session_state.get('gui_initialized', False)}")
            st.info(f"Session started: {st.session_state.get('session_started', False)}")
            if hasattr(self, 'random_id'):
                st.info(f"Current session ID: {self.random_id}")
        
        st.title("🧬 Protify - Protein Analysis Platform")
        
        # Build sidebar and get selected page
        page = self.build_sidebar()
        
        # Route to appropriate page
        if page == "Info":
            self.build_info_page()
        elif page == "Model":
            self.build_model_page()
        elif page == "Data":
            self.build_data_page()
        elif page == "Embedding":
            self.build_embedding_page()
        elif page == "Probe":
            self.build_probe_page()
        elif page == "Trainer":
            self.build_trainer_page()
        elif page == "Scikit":
            self.build_scikit_page()
        elif page == "Replay":
            self.build_replay_page()
        elif page == "Visualization":
            self.build_viz_page()
        
        # Show current status in sidebar
        with st.sidebar:
            st.markdown("---")
            st.subheader("Status")
            
            # Session status
            if st.session_state.session_started:
                if hasattr(self, 'random_id'):
                    st.success(f"✓ Session: {self.random_id}")
                else:
                    st.success("✓ Session started")
            else:
                st.warning("⚠ Session not started")
            
            # Task status
            if st.session_state.task_running:
                st.warning("🔄 Task running...")
            
            # Progress indicators
            progress_items = [
                ("Models selected", st.session_state.models_selected),
                ("Data loaded", st.session_state.data_loaded),
                ("Embeddings saved", st.session_state.embeddings_saved),
                ("Probe configured", st.session_state.probe_args_created)
            ]
            
            for item_name, completed in progress_items:
                if completed:
                    st.success(f"✓ {item_name}")
                else:
                    st.info(f"○ {item_name}")


def main():
    gui = StreamlitGUI()
    gui.run()


if __name__ == "__main__":
    main() 