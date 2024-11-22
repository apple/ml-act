# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import logging
import typing as t
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import act.hooks as hooks
from act.datasets import get_dataloader, get_dataset
from act.datasets.responses_io import ResponsesLoader
from act.models import get_model
from act.models.model_with_hooks import get_model_with_hooks
from act.utils import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ResponsesManager:
    """Computes and saves responses from specified modules within a pretrained language model.

    This class handles loading a pretrained model, setting up custom hooks to extract responses from desired modules,
    and iterating over a dataset to compute and save the extracted responses.

    Attributes:
        cfg (DictConfig): Configuration object containing model parameters, task settings, data paths, and intervention details.
        output_path (Path): Path where computed responses will be saved.
        model (nn.Module): Loaded pretrained language model with registered hooks for response extraction.
        module_names (List[str]): List of module names from which to extract responses.

    Args:
        cfg (DictConfig): Configuration object defining the model, task, and intervention parameters.

    Example usage:

    ```python
    # Assuming cfg is a loaded DictConfig object
    responses_manager = ResponsesManager(cfg)
    output_path = responses_manager.compute_responses()
    print(f"Responses saved to: {output_path}")
    ```

    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.output_path = self.get_output_path(cfg)

        assert cfg.device != "cuda" or torch.cuda.is_available()
        logger.info(f"Loading model to extract responses.")
        # Models and Tokenizers
        module, tokenizer = get_model(
            cache_dir=cfg.cache_dir,
            device=cfg.device,
            rand_weights=False,
            model_task=cfg.task_params.model_task,
            **cfg.model_params,
        )
        logger.info(f"Done loading model.")
        self.model = get_model_with_hooks(
            module=module,
            model_task=cfg.task_params.model_task,
            device=cfg.device,
            **cfg.model_params,
        )

        # Datasets
        train_dataset, collate_fn = get_dataset(
            name=cfg.task_params.dataset,
            datasets_folder=Path(cfg.data_dir),
            split="train",
            subsets=list(cfg.task_params.src_subsets + cfg.task_params.dst_subsets),
            tokenizer=tokenizer,
            **cfg.task_params.get("dataset_params", {}),
        )

        self.module_names = self.get_module_names(cfg.model_params.module_names)

        # Sampling and dataloader
        self.dataloader = get_dataloader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=cfg.shuffle,
            balanced=cfg.balanced_data,
            seed=cfg.seed,
        )

    def get_module_names(self, module_names: t.List[str]):
        """Get the names of modules in the model.

        Args:
            module_names (list of str): A regex list of module names.

        Returns:
            None
        """
        return self.model.find_module_names(self.model.module, module_names)

    def _set_hooks(self, module_names: t.List[str], extra_hooks: list = None) -> None:
        """Registers hooks to extract responses from specified modules and save them.

        This method iterates through the list of desired module names (`module_names`) and registers a forward hook on each module.
        The hook function, defined using `hooks.get_hook`, captures the output of the module during the forward pass
        and saves it to disk along with other relevant metadata specified in `self.cfg.save_fields`.

        The `pooling_op` parameter determines how the outputs from different modules are aggregated.  It can be a single string
        representing a pooling operation (e.g., "mean", "max") or a list of strings for more complex aggregation schemes.

        Args:
            module_names (t.List[str]): A list of module names within the model from which to extract responses.
            extra_hooks (list, optional): A list of additional hook functions to register. Defaults to None.

        Raises:
            KeyError: If a specified module name does not exist within the model.

        """
        if isinstance(self.cfg.intervention_params.pooling_op, str):
            pooling_op = [self.cfg.intervention_params.pooling_op]
        else:
            pooling_op = self.cfg.intervention_params.pooling_op
        hook_fns = [
            hooks.get_hook(
                "postprocess_and_save",
                module_name=module_name,
                pooling_op_names=pooling_op,
                output_path=self.output_path,
                save_fields=[
                    "id",
                ]
                + self.cfg.save_fields,
                threaded=False,
                **self.cfg.intervention_params.hook_params,
            )
            for module_name in module_names
        ]
        if extra_hooks is not None:
            hook_fns += extra_hooks
        self.model.remove_hooks()
        self.model.register_hooks(hook_fns)

    @staticmethod
    def get_output_path(cfg: DictConfig) -> Path:
        """Returns the output path for the computed responses."""
        # Sanity check, only allowing multiplicity for args.module_names and args.subset
        model_name = Path(cfg.model_params.model_path).name
        # Setting paths
        return Path(cfg.cache_dir) / cfg.tag / model_name / cfg.task_params.dataset

    def compute_responses(
        self, module_names: t.List[str] = None, extra_hooks: list = []
    ) -> Path:
        """Computes and saves responses from specified modules in the model.

        This method orchestrates the process of extracting responses from designated modules within the model during inference. It handles hook registration, batch processing, and saving the extracted responses to disk.

        Args:
            module_names (List[str], optional): List of module names from which to extract responses. If None, defaults to the configured module names (`self.module_names`).
            extra_hooks (list, optional): Additional hook functions to register during computation. Defaults to an empty list.

        Returns:
            Path: The path where the computed responses are saved.

        """
        if module_names is None:
            module_names = self.module_names

        self._set_hooks(module_names, extra_hooks)
        logger.info(f"Computing responses for {len(module_names)} modules.")
        logger.debug("\n".join(module_names))

        current_batch = 0
        max_batches = (
            len(self.dataloader)
            if self.cfg.max_batches is None
            else min(len(self.dataloader), self.cfg.max_batches)
        )
        if current_batch == self.cfg.max_batches:
            logger.info("All batches found, nothing to compute")
        else:
            if current_batch > 0:
                logger.info(f"Resuming from batch {current_batch}")
            else:
                logger.info("Computing batch responses")

            iloader = iter(self.dataloader)
            for idx in tqdm(range(max_batches), desc="Computing responses"):
                batch = next(iloader)
                if idx >= current_batch:
                    with torch.inference_mode():
                        self.model.update_hooks(batch_idx=idx, batch=batch)
                        try:
                            self.model(batch)
                        except hooks.custom_exceptions.TargetModuleReached:
                            pass
                # checkpoint["current_batch"] = idx + 1
                # TODO Save hooks state if stateful
                # torch.save(checkpoint, checkpoint_path)
            logger.info("Done")
        return self.output_path


class InterventionsManager:
    """Manages the learning and application of interventions on a generative model.

    This class handles loading model responses, configuring intervention parameters,
    and learning intervention strategies for specific modules within the model.

    Interventions are techniques used to modify the behavior of a model by injecting biases
    or constraints during inference. They can be used for tasks like domain adaptation,
    style transfer, or controlling the output of a model.

    Attributes:
        responses_path (Path): The path to the directory where model responses are stored.
        cfg (DictConfig): A configuration object containing parameters for the interventions manager.

    """

    def __init__(self, responses_path: Path, cfg: DictConfig):
        """Initialize the InterventionsManager.

        Args:
            responses_path (Path): The path to the directory containing model response data.
            cfg (DictConfig): A configuration object defining intervention parameters and settings.

        Returns:
            None
        """
        self.responses_path = responses_path
        self.cfg = cfg

        self.output_path = self.get_output_path(cfg)

    @staticmethod
    def get_output_path(cfg: DictConfig) -> Path:
        """Determine the output directory for storing intervention data.

        Args:
            cfg (DictConfig): The configuration object containing intervention parameters.

        Returns:
            Path: The path to the directory where intervention results will be saved.
        """
        model_name = Path(cfg.model_params.model_path).name
        sorted_keys = list(sorted(cfg.intervention_params.keys()))
        name = "-".join(
            [
                cfg.intervention_params[k]
                for k in sorted_keys
                if k not in ["hook_params", "state_path"]
            ]
        )
        return Path(cfg.cache_dir) / cfg.tag / model_name / name

    def _learn_intervention(self, module_name: str):
        """Learn an intervention strategy for a specific module.

        This method loads relevant model responses, applies the intervention to those responses,
        and learns the parameters of the intervention strategy based on the modified data.

        Args:
            module_name (str): The name of the module within the language model for which the intervention will be learned.

        Returns:
            None
        """
        # Load responses for a given module
        data_subset = self.responses_loader.load_data_subset(
            {
                "module_names": [module_name],
                "pooling_op": [self.cfg.intervention_params.pooling_op],
                "subset": list(
                    self.cfg.task_params.src_subsets + self.cfg.task_params.dst_subsets
                ),
            },
            num_workers=0,
        )
        data_subset = ResponsesLoader.label_src_dst_subsets(
            data_subset,
            src_subset=self.cfg.task_params.src_subsets,
            dst_subset=self.cfg.task_params.dst_subsets,
            balanced=True,
            seed=self.cfg.seed,
        )
        z = torch.tensor(data_subset["responses"])
        y = torch.tensor(data_subset["label"]).to(torch.bool)

        # Create hook object with requested hook args
        hook = hooks.get_hook(
            self.cfg.intervention_params.name,
            module_name=module_name,
            **self.cfg.intervention_params.hook_params,
        )

        # Estimate hook parameters
        hook.fit(
            responses=z,
            labels=y,
            **self.cfg.intervention_params.hook_params,
        )

        # Save hook
        self.output_path.mkdir(exist_ok=True, parents=True)
        hook.save_state_dict(state_path=self.output_path / (module_name + ".statedict"))

    def learn_intervention_all(self) -> None:
        """Learn interventions on all modules specified in the configuration.

        Args:
            None

        Returns:
            None
        """
        logger.info("Learning all interventions at once.")
        self.responses_loader = ResponsesLoader(
            root=self.responses_path,
            from_folders=[
                f"*/*/{self.cfg.intervention_params.pooling_op}",
            ],
            columns=["responses", "id", "subset"] + self.cfg.load_fields,
        )
        # Get only the module names requested via args
        module_names = self.responses_loader.get_attribute_values(
            "module_names", filter_patterns=self.cfg.model_params.module_names
        )
        for module_name in module_names:
            self._learn_intervention(module_name=module_name)

    def learn_intervention_all_incremental(self, responses_manager: ResponsesManager):
        """Learn interventions incrementally for all specified modules.

        This method allows for learning interventions on a module-by-module basis,
        leveraging previously learned interventions from previous modules when learning a new one.

        Args:
            responses_manager (ResponsesManager): A manager to handle and load responses from files.

        Returns:
            None
        """
        logger.info("Learning interventions incrementally")
        # Get only the module names requested via args
        module_names = responses_manager.get_module_names(
            module_names=self.cfg.model_params.module_names
        )
        # Here we load intervention hooks for all those layers already computed (used_module_names).
        used_module_names = []
        for module_name in module_names:
            intervention_hooks = []
            for used in used_module_names:
                state_path = (
                    Path(self.cfg.cache_dir) / self.output_path / f"{used}.statedict"
                )
                # These hooks will intervene on the model, that's why we need hook_* args.
                hook = hooks.get_hook(
                    self.cfg.intervention_params.name,
                    module_name=used,
                    **self.cfg.intervention_params.hook_params,
                )
                hook.from_state_path(state_path)
                intervention_hooks.append(hook)
            logger.info(
                f"Loaded {len(intervention_hooks)} hooks: [{''.join(used_module_names[:1])}:{''.join(used_module_names[-1:])}]"
            )
            # Register all the hooks.
            logger.info(f"New inference on {module_name}.")

            # 1. Save responses for current module
            responses_manager.compute_responses(
                [module_name], extra_hooks=intervention_hooks
            )
            # 2. Learn intervention for current module
            # 2.1. Search directory tree for new responses
            self.responses_loader = ResponsesLoader(
                root=self.responses_path,
                from_folders=[
                    f"*/*/{self.cfg.intervention_params.pooling_op}",
                ],
                columns=["responses", "id", "subset"] + self.cfg.load_fields,
            )
            # 2.2. Learn intervention
            self._learn_intervention(module_name)
            # 3. Finally add the current module name in the used ones
            used_module_names.append(module_name)


# Fix to cast dtypes in yaml config
OmegaConf.register_new_resolver(
    name="dtype", resolver=lambda dtype: hydra.utils.get_object(dtype)
)


@hydra.main(config_path="../configs", config_name="text_generation")
def main(cfg: DictConfig) -> t.Optional[InterventionsManager]:
    return learn_intervention(cfg)


def learn_intervention(cfg: DictConfig) -> t.Optional[InterventionsManager]:
    """Learn interventions on a model from a given configuration.

    Args:
        cfg (DictConfig): The configuration parameters for the intervention learning process.

    Returns:
        InterventionsManager: An instance of the InterventionsManager class if any interventions were learned, otherwise None.
    """
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=False)}")

    # Set random seed
    utils.seed_all(cfg.seed)
    if cfg.interventions.intervention_params.incremental == "incr":
        cfg.responses.intervention_params.hook_params.raise_exception = True
        logger.warning(
            "Responses hook will raise exception to speedup incremental training."
        )
    responses_manager = ResponsesManager(cfg.responses)
    interventions_manager = InterventionsManager(
        ResponsesManager.get_output_path(cfg.responses), cfg.interventions
    )

    if cfg.interventions.intervention_params.incremental == "atonce":
        if cfg.compute_responses:
            responses_manager.compute_responses()
        interventions_manager.learn_intervention_all()
    elif cfg.interventions.intervention_params.incremental == "incr":
        if not cfg.compute_responses:
            raise RuntimeError(
                "Cannot learn incremental interventions with cfg.compute_responses=false."
            )
        interventions_manager.learn_intervention_all_incremental(responses_manager)
    else:
        raise ValueError(f"Unknown incremental mode {cfg.interventions.incremental}")
    return interventions_manager


if __name__ == "__main__":
    main()
