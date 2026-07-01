import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Set

import kagglehub
import nbformat
from kagglesdk.kaggle_object import KaggleObject, TimeDeltaSerializer
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelRequest,
    ApiSaveKernelRequest,
)
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from omegaconf import OmegaConf

from mxsep.cfg import Config
from mxsep.kaggle import KaggleStore
from mxsep.kaggle.utils import create_id_from_config

## START Monkey-patch TimeDeltaSerializer._from_dict_value

original_from_dict = TimeDeltaSerializer._from_dict_value

def patched_from_dict_value(value):
    try:
        # Try to parse with the original logic first
        return original_from_dict(value)
    except ValueError:
        # Handle the case where there's no decimal point
        value_str = value.rstrip('s')
        if '.' not in value_str:
            # Only seconds present
            seconds = int(value_str)
            return timedelta(seconds=seconds)
        # If there is a decimal but parsing failed, re-raise
        raise

# Apply the patch
TimeDeltaSerializer._from_dict_value = staticmethod(patched_from_dict_value)

## END Monkey-Patch

def create_notebook(user: str, cmd: str, cfg: Config):
    """

    :param kernel_ref: Kernel ref in format <owner>/<kernel-slug>
    :param cmd: script cmd to run. ex: 'mxsep-train'
    :param cfg: config for running script
    """
    run_id = create_id_from_config(OmegaConf.to_container(cfg, resolve=False))
    print(f"Run ID: {run_id}")
    kaggle_execution_id = 0
    gpu_quota = get_quota_gpu(user)
    kaggle_gpu_max_run_time = 12 * 3600
    overhead = 1800 # overhead loading dataset...
    max_runtime = min(gpu_quota, kaggle_gpu_max_run_time)
    max_runtime = max_runtime - overhead
    cfg.training.max_runtime = max_runtime

    slug = f'mxsep-{run_id}-{kaggle_execution_id}'
    cfg.training.monitoring.wandb.run_id = run_id
    
    kernel_ref = f"{user}/{slug}"
    store = KaggleStore().load_store()
    kaggle_data_sources = find_kaggle_data_sources(cfg)

    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=False)

    # Create a new notebook
    nb = new_notebook()

    nb.metadata = {
        "kernelspec": {
            "language": "python",
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.13",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    }

    # Install dependencies
    nb.cells.append(
        new_code_cell("%pip install git+https://github.com/joris-vaneyghen/mxsep.git")
    )
    nb.cells.append(
        new_code_cell("%pip install dotenv")
    )
    if store.dataset_env:
        # load env
        kaggle_data_sources["datasets"].add(store.dataset_env)
        nb.cells.append(
            new_code_cell(
                f"%load_ext dotenv\n%dotenv /kaggle/input/datasets/{store.dataset_env}/.env -o"
            )
        )

    nb.cells.append(new_code_cell("%mkdir my_configs"))

    nb.cells.append(new_code_cell(f"%%writefile my_configs/my_config.yaml\n{cfg_yaml}"))

    nb.cells.append(
        new_code_cell(f"!{cmd} --config-dir my_configs --config-name my_config")
    )

    


    source_code = nbformat.writes(nb)
    save_and_run_all(
        source_code=source_code,
        slug=slug,
        user=user,
        kaggle_data_sources=kaggle_data_sources,
    )

    # OLD IMPL
    # todo remove

    # path = Path(f"./kaggle/{kernel_ref}")
    # path.mkdir(parents=True, exist_ok=True)
    #
    # notebook_file = f"{slug}.ipynb"
    # notebook_path = path / notebook_file
    # metadata_path = path / "kernel-metadata.json"

    # # Save the notebook
    # with open(notebook_path, "w", encoding="utf-8") as f:
    #     nbformat.write(nb, f)

    # metadata = {
    #     "id": kernel_ref,
    #     "title": slug.replace("-", " "),
    #     "code_file": notebook_file,
    #     "language": "python",
    #     "kernel_type": "notebook",
    #     "is_private": True,
    #     "enable_gpu": True,
    #     "enable_tpu": False,
    #     "enable_internet": True,
    #     "machine_shape": "NvidiaTeslaT4",
    #     "dataset_sources": list(kaggle_data_sources['datasets']),
    #     "competition_sources": [],
    #     "kernel_sources": list(kaggle_data_sources['notebooks']),
    #     "model_sources": list(kaggle_data_sources['models']),
    # }
    #
    # with open(metadata_path, "w", encoding="utf-8") as f:
    #     json.dump(metadata, f)


def append_kaggle_data_source(
    kaggle_data_sources: Dict[str, Set[str]], path: str
) -> None:
    """
    Extract Kaggle source information from a path and add it to the appropriate set.

    Args:
        kaggle_data_sources: Dictionary containing sets for 'datasets', 'notebooks', and 'models'
        path: string to parse

    Returns:
        None (modifies kaggle_data_sources in place)
    """
    # Convert to string and normalize
    path_str = str(path)

    # Patterns for different Kaggle input types
    # Pattern: /kaggle/input/datasets/{user}/{slug}/*
    datasets_pattern = r"/kaggle/input/datasets/([^/]+)/([^/]+)/"
    # Pattern: /kaggle/input/notebooks/{user}/{slug}/*
    notebooks_pattern = r"/kaggle/input/notebooks/([^/]+)/([^/]+)/"
    # Pattern: /kaggle/input/models/{user}/{slug}/{framework}/{variation}/{version}/*
    models_pattern = r"/kaggle/input/models/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/"

    # Check datasets pattern
    match = re.search(datasets_pattern, path_str)
    if match:
        user, slug = match.groups()
        kaggle_data_sources["datasets"].add(f"{user}/{slug}")
        return

    # Check notebooks pattern
    match = re.search(notebooks_pattern, path_str)
    if match:
        user, slug = match.groups()
        kaggle_data_sources["notebooks"].add(f"{user}/{slug}")
        return

    # Check models pattern
    match = re.search(models_pattern, path_str)
    if match:
        user, slug, framework, variation, version = match.groups()
        kaggle_data_sources["models"].add(
            f"{user}/{slug}/{framework}/{variation}/{version}"
        )
        return


def find_kaggle_data_sources(cfg: Config) -> Dict[str, Set[str]]:
    """
    Find all Kaggle source references in the configuration. (For a source of type notebook we recursive append also it's sources)

    Args:
        cfg: Configuration object containing paths that might reference Kaggle sources

    Returns:
        Dictionary with sets of unique Kaggle source identifiers
    """
    kaggle_data_sources = {
        "datasets": set(),
        "notebooks": set(),
        "models": set(),
    }

    # Helper function to safely append paths
    def safe_append_paths(paths):
        """Safely append paths, handling both single paths and iterables."""
        if paths is None:
            return
        if isinstance(paths, (str, Path)):
            append_kaggle_data_source(kaggle_data_sources, str(paths))
        elif isinstance(paths, (list, tuple, set)):
            for path in paths:
                append_kaggle_data_source(
                    kaggle_data_sources, str(path) if path else None
                )
        else:
            # Try to handle as single value
            append_kaggle_data_source(kaggle_data_sources, str(paths))

    # Check main configuration paths
    safe_append_paths(cfg.training.resume_from_checkpoint)
    safe_append_paths(cfg.dataset.train.predefined_jsonl_path)
    safe_append_paths(cfg.dataset.validation.predefined_jsonl_path)

    # Check audio file CSVs
    safe_append_paths(cfg.dataset.train.audio_file_csv)
    safe_append_paths(cfg.dataset.validation.audio_file_csv)

    # Check audio file patterns
    safe_append_paths(cfg.dataset.train.audio_files_pattern)
    safe_append_paths(cfg.dataset.validation.audio_files_pattern)

    notebook_refs = kaggle_data_sources["notebooks"].copy()
    for notebook_ref in notebook_refs:
        recursive_lookup_kaggle_data_sources(kaggle_data_sources, notebook_ref)

    return kaggle_data_sources


def recursive_lookup_kaggle_data_sources(
    kaggle_data_sources: Dict[str, Set[str]], ref: str
):
    kaggle_client = kagglehub.http_resolver.build_kaggle_client()
    api = kaggle_client.kernels.kernels_api_client
    request = ApiGetKernelRequest()
    request.user_name = ref.split("/")[0]
    request.kernel_slug = ref.split("/")[1]
    response = api.get_kernel(request=request)
    kaggle_data_sources["datasets"].update(response.metadata.dataset_data_sources)
    kaggle_data_sources["models"].update(response.metadata.model_data_sources)
    for notebook_ref in response.metadata.kernel_data_sources:
        if notebook_ref not in kaggle_data_sources["notebooks"]:
            kaggle_data_sources["notebooks"].add(notebook_ref)
            recursive_lookup_kaggle_data_sources(kaggle_data_sources, notebook_ref)


def save_and_run_all(
    source_code: str, user: str, slug: str, kaggle_data_sources: Dict[str, Set[str]]
):
    # todo choose
    print(f"Kaggle logged in as {kagglehub.whoami()}")
    kaggle_client = kagglehub.http_resolver.build_kaggle_client()
    api = kaggle_client.kernels.kernels_api_client
    request = ApiSaveKernelRequest()
    request.slug = f"{user}/{slug}"
    request.new_title = slug.replace("-", " ")
    request.text = source_code
    request.language = "python"
    request.kernel_type = "notebook"
    request.dataset_data_sources = list(kaggle_data_sources["datasets"])
    request.kernel_data_sources = list(kaggle_data_sources["notebooks"])
    request.model_data_sources = list(kaggle_data_sources["models"])
    request.competition_data_sources = []
    request.category_ids = ['gpu']
    request.is_private = False
    request.enable_gpu = True #deprecated
    request.enable_tpu = False #deprecated
    request.enable_internet = True
    request.machine_shape = "NvidiaTeslaT4"

    # For  P100
    # request.machine_shape = "Gpu" #P100
    # see https://github.com/Kaggle/docker-python/releases https://console.cloud.google.com/artifacts/docker/kaggle-gpu-images
    # use docker v159 with torch 2.5.1+cu124 (compatible with P100 - SM 6.0)
    # request.docker_image_pinning_type = 'original'
    # request.docker_image = 'gcr.io/kaggle-private-byod/python@sha256:5b3264d2d3aefa4f5e4d6f036b0d2b2d9691dab62493ae3af85a3153608f30a7'

    request.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
    response = api.save_kernel(request=request)
    print(response.to_dict())



def get_quota_gpu(user: str)->int:
    # todo choose user
    print(f"Kaggle logged in as {kagglehub.whoami()}")
    kaggle_client = kagglehub.http_resolver.build_kaggle_client()
    api = kaggle_client.kernels.kernels_api_client
    response = api.get_accelerator_quota_statistics()
    time_allowed = response.gpu_quota.total_time_allowed.total_seconds()
    time_used = response.gpu_quota.time_used.total_seconds()
    return int(time_allowed - time_used)