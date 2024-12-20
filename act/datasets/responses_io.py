# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import logging
import os
import re
import typing as t
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ResponsesLoader:
    """
    A utility class for loading and managing sets of responses generated by foundation models.
    It provides functionality to parse directory structures, identify unique attribute values,
    and prepare data for further processing or analysis.

    **Attributes**

    * **ORDERED_ATTRIBUTES** (list): The expected order of attributes in the directory structure.
      Default: `["tag", "model_name", "dataset", "subset", "module_names", "pooling_op"]`
    * **root** (Path): The root directory where all responses are stored.
    * **attribute_values** (dict): A dictionary containing unique values for each attribute,
      updated during the parsing process. Initializes with empty sets for each attribute.
    * **file_format** (str): The default file format for loading responses. Default: `*.pt`
    * **columns** (t.List[str], optional): Specific columns to focus on when loading data. Defaults to None.

    **Methods**

    * `__init__(root, from_folders=None, file_format='*.pt', columns=None)`: Initializes the ResponsesLoader object.
    * `parse_folder(folder)`: Parses the response folder structure for unique attribute values and prepares data.

    **Initialization Example**

    ```python
        from act.datasets.responses_io import ResponsesLoader
        from pathlib import Path

        # Specify the root directory and optional folders to parse
        root_dir = Path('/path/to/responses')
        from_folders = ["*/gpt2/*/*/*/mean"] # Following ORDERED_ATTRIBUTES

        # Initialize the ResponsesLoader with custom file format and columns (if needed)
        loader = ResponsesLoader(root_dir, from_folders=folders_to_parse,
                                file_format='*.pt', columns=['responses', 'module_names'])
    ```
    """

    ORDERED_ATTRIBUTES = [
        "tag",
        "model_name",
        "dataset",
        "subset",
        "module_names",
        "pooling_op",
    ]

    def __init__(
        self,
        root: Path,
        from_folders: t.List[Path] = None,
        file_format: str = "*.pt",
        columns: t.List[str] = None,
    ):
        self.root = Path(root)
        self.attribute_values = {k: set([]) for k in self.ORDERED_ATTRIBUTES}
        self.file_format = file_format
        self.columns = columns

        if from_folders is not None:
            self.file_trees = self.parse_folder(from_folders)
            logger.info("Parsed directory tree. Found:")
            for k, v in self.attribute_values.items():
                v = list(v)
                if len(v) > 2:
                    logger.info(f"{k} (count={len(v)})")
                    logger.debug("\n".join(v))
                else:
                    logger.info(f"{k} (count={len(v)}): {str(v)}")

    def _parse_folder_tree(self, root: Path, path_parts: t.List) -> dict:
        """Recursive Folder Tree Parser

        Traverses a directory structure, identifying unique attribute values based on the predefined `ORDERED_ATTRIBUTES` order.
        This method is used to construct a dictionary of file paths and attribute values, facilitating data organization.

        **Parameters**

        * **root** (Path): The base directory from which to initiate the parsing process.
        * **path_parts** (list[str]): A list of subdirectory names (in string format) representing the path to parse,
                                    relative to the specified `root`. These parts are mapped to attributes in
                                    `ORDERED_ATTRIBUTES`.

        **Returns**

        * **dict**: A nested dictionary containing:
            + **Keys**: Unique attribute values found within the parsed directories.
            + **Values**:
                - If final directory level (i.e., `len(path_parts) == 0`): A list of file paths matching `self.file_format`.
                - Otherwise: Another dictionary representing the next level of attributes and/or files.

        **Side Effects**

        * **Updates `self.attribute_values`**: With unique attribute names discovered during directory traversal,
                                            organized by their corresponding attribute name in `ORDERED_ATTRIBUTES`.

        **Example Directory Structure and Usage**

        Suppose we have:
        ```
        root/
        |-- tag_A
        |   |-- model_name_1
        |   |   |-- dataset_X
        |   |   |   |-- file1.pt
        |   |   |-- dataset_Y
        |   |-- model_name_2
        |-- tag_B
            |-- ...
        ```python
            loader = ResponsesLoader(root='/path/to/root')
            path_parts = ['tag_A', 'model_name_1', 'dataset_X']
            parsed_result = loader._parse_folder_tree(loader.root, path_parts)
            print(parsed_result)  # Output: {'file1.pt'} (assuming file_format='*.pt')
            print(loader.attribute_values)
            # Output: {'tag': {'tag_A'}, 'model_name': {'model_name_1'}, 'dataset': {'dataset_X'}}
        ```
        """
        if len(path_parts) == 0:
            return list(root.glob(self.file_format))
        ret = {}
        attribute_name = self.ORDERED_ATTRIBUTES[-len(path_parts)]
        current_part = path_parts[0]
        part_paths = list(root.glob(current_part))
        for part_path in part_paths:
            if os.path.isdir(part_path):
                self.attribute_values[attribute_name].add(part_path.name)
                ret[str(part_path.name)] = self._parse_folder_tree(
                    part_path, path_parts[1:]
                )
        return ret

    def _filter_tree(self, tree, level: int = 0, filter: t.Dict = {}):
        """Recursive Tree Filtering

        Applies attribute-based filtering to a nested dictionary (`tree`) representing a directory structure.
        This method traverses the tree, selectively including branches that match specified patterns at each level,
        as defined by `ORDERED_ATTRIBUTES`.

        **Parameters**

        * **tree** (dict): The nested dictionary to be filtered, where keys represent attribute values and
                        values are either sub-dictionaries or leaf nodes (e.g., file paths).
        * **level** (int, optional): The current recursion depth, corresponding to an index in `ORDERED_ATTRIBUTES`.
                                    Defaults to 0 (the first attribute).
        * **filter** (dict[str, str or iterable[str]], optional): A dictionary specifying filtering patterns.
            - **Keys**: Attribute names that must match an entry in `ORDERED_ATTRIBUTES` (at the current `level` or any subsequent level).
            - **Values**: Patterns to match (as strings or iterables of strings) for the corresponding attribute values.
                        Patterns are interpreted as regular expressions. If a value is not a list/tuple/set, it's treated as a single pattern.
                        Omitting an attribute from this dictionary implies no filtering at that level (`.*` matches all).

        **Returns**

        * **dict**: The filtered tree, with only branches matching the provided patterns at each relevant level.

        **Example Filtering**

        Given:
        ```python
            tree = {
                'tag_A': {
                    'model_name_1': {'dataset_X': ['file1.pt'], 'dataset_Y': ['file2.pt']},
                    'model_name_2': {'dataset_Z': ['file3.pt']}
                },
                'tag_B': {...}
            }

            filter_specs = {
                'tag': ['tag_A'],
                'dataset': ['dataset_X', 'dataset_Z']
            }
            filtered_tree = loader._filter_tree(tree, filter=filter_specs)
            print(filtered_tree)
            Output: {'tag_A': {'model_name_1': {'dataset_X': ['file1.pt']}, 'model_name_2': {'dataset_Z': ['file3.pt']}}}
        ```
        **Note**: This is an internal helper method. Typically, you would interact with the `ResponsesLoader` class through its public methods.
        """

        if level == len(self.ORDERED_ATTRIBUTES):
            return tree
        new_tree = {}
        attribute_name = self.ORDERED_ATTRIBUTES[level]
        patterns = filter.get(attribute_name, [".*"])
        if not isinstance(patterns, (list, tuple, set)):
            patterns = [patterns]
        all_keys = list(tree.keys())
        filtered_keys = []
        for key in all_keys:
            if any([re.fullmatch(pattern, key) is not None for pattern in patterns]):
                filtered_keys.append(key)
        if not isinstance(filtered_keys, (list, tuple)):
            filtered_keys = [filtered_keys]
        for key in filtered_keys:
            new_tree[key] = self._filter_tree(tree[key], level + 1, filter=filter)
        return new_tree

    def _flatten_tree(
        self, tree: dict, level: int = 0, attributes: dict = {}
    ) -> t.List[dict]:
        """
        Filter a parsed folder tree based on provided attributes.

        Args:
            tree (dict): A dictionary representing the parsed folder tree to be filtered.

            level (int): The current depth of recursive parsing or filtering, 0 by default.

            filter (dict): An optional dictionary specifying which keys in `tree` should be included based on attributes.
                        If an attribute is not specified in the filter, all unique values for that attribute are used.
                        E.g., {"subset": ["dogs", "cats"], "pooling_op": "mean"} will only include these two data subsets even if more exist.

        Returns:
            dict: A filtered version of the input tree, with only nodes matching provided filters retained.

        Note:
            The filtering operation applies to the `ORDERED_ATTRIBUTES` order, and includes keys in the output if they match
            any of the specified values for that attribute (or all unique values if no filter is provided).
        """
        ret = []
        if level == len(self.ORDERED_ATTRIBUTES):
            assert isinstance(tree, (list, tuple))
            for v in tree:
                new_attributes = deepcopy(attributes)
                new_attributes["path"] = v
                ret.append(new_attributes)
            return ret
        for k, v in tree.items():
            attribute_name = self.ORDERED_ATTRIBUTES[level]
            new_attributes = deepcopy(attributes)
            new_attributes[attribute_name] = k
            ret.extend(self._flatten_tree(v, level + 1, new_attributes))
        return ret

    def get_attribute_values(
        self, attribute_name: str, filter_patterns: t.Optional[t.List[str]] = None
    ) -> set:
        """
        Retrieve unique values of a specific attribute across all parsed directories.

        Args:
            attribute_name (str): The name of the attribute to retrieve unique values for.
            filter_patterns (list): Return only those values that match the filters (glob mode).
        Returns:
            set: A set containing unique values for the specified attribute in all parsed directories.

        Raises:
            AssertionError: If `attribute_name` is not one of the ORDERED_ATTRIBUTES.

        Note:
            The uniqueness of attribute values across all parsed directories are stored and maintained
            in the `attribute_values` dictionary during parsing operations.
        """
        import re

        assert attribute_name in self.ORDERED_ATTRIBUTES
        if filter_patterns:
            _filter_patterns = []
            # Module names might end in ":[0-9]+"
            if attribute_name == "module_names":
                for idx, pattern in enumerate(filter_patterns):
                    if ":" not in pattern:
                        _filter_patterns.append(f"{pattern}(:[0-9]+)?")
                    else:
                        _filter_patterns.append(pattern)

        else:
            _filter_patterns = [".*"]
        ret = set()
        values = self.attribute_values[attribute_name]
        for f in _filter_patterns:
            filtered_values = set(
                filter(lambda value: re.fullmatch(f, value) is not None, values)
            )
            ret.update(filtered_values)
        return ret

    def parse_folder(self, folders: t.List[Path]) -> t.List[dict]:
        """
        Parse a response folder structure for unique values of attributes and prepare it for data loading.

        Args:
            folder (str): The a list of directories to start parsing from as Paths / strings.

        Returns:
            dict: A dictionary representing the parsed folder tree, with paths for files matching
                file_format and unique values of attributes.

        Side effects:
            Updates attribute_values with unique names found within directories in 'folder'.

        Note:
            The parsing operation applies to the `ORDERED_ATTRIBUTES` order, and stores all unique
            combinations of the first len(ORDERED_ATTRIBUTES) parts of paths.
        """
        ret = []
        if not isinstance(folders, (list, tuple)):
            folders = [folders]
        folders = map(Path, folders)
        for folder in folders:
            new_root = Path(*self.root.parts[: -len(folder.parts)])
            new_folder = Path(*self.root.parts[-len(folder.parts) :]) / folder
            ret.append(self._parse_folder_tree(new_root, new_folder.parts))
        return ret

    def load_data_subset(
        self,
        attribute_values: t.Optional[t.Dict[str, t.List[str]]] = None,
        batch_size: int = 128,
        num_workers: int = 0,
        multi_output: bool = True,
    ) -> dict:
        """
        Load a subset of responses based on provided attributes and prepare it for further processing.

        Args:
            attribute_values (dict): An optional dictionary specifying which keys in the filtered tree
                                    should be included as per their respective attributes. If an attribute
                                    is not specified, all unique values for that attribute are used.
                                    e.g. {"pooling_op": ["mean"]}}

            batch_size (int): The number of samples to load at a time, default is 128.

            num_workers (int): The number of subprocesses to use for data loading, default is 0.
            multi_output (bool): Whether to match all layers ending with :[0-9]* when no particular ":" has been specified.

        Returns:
            dict: A dictionary containing loaded responses, grouped by attributes and batch-wise. Each key in the
                returned dictionary corresponds to an attribute from `ORDERED_ATTRIBUTES`. The values are lists of
                tensors corresponding to each unique combination of attribute values across batches.

        Note:
            Loading operation applies a filter based on provided `attribute_values` and loads responses in batch-wise,
            allowing for parallelized data loading with the help of PyTorch's DataLoader. The returned dictionary can
            be used directly for further processing or analysis tasks.

        Side effects:
            Updates attribute_values based on provided filters if any.
        """
        torch.multiprocessing.set_sharing_strategy("file_system")
        logger.info(
            f"Loading data subset: {str(attribute_values) if len(attribute_values) > 0 else 'all_data'}"
        )
        # Making sure the passed keys are valid keys according to the current schema.
        assert all(
            [k in self.ORDERED_ATTRIBUTES for k in attribute_values.keys()]
        ), f"Some of the keys in {attribute_values.keys()} are not valid.\nValid keys are {self.ORDERED_ATTRIBUTES}."
        # Making sure the passed values are lists. This removes some magic and prevents silent bugs
        assert all(
            isinstance(v, list) for v in attribute_values.values()
        ), f"{attribute_values} contains non-iterable values."
        if multi_output and "module_names" in attribute_values:
            module_names = []
            for idx, module_name in enumerate(attribute_values["module_names"]):
                if ":" not in module_name:
                    # Matches anything ending with :[0-9]+ or ending with "".
                    module_names.append(f"{module_name}(:[0-9]+)?")
                else:
                    module_names.append(module_name)
            attribute_values["module_names"] = module_names

        # No None values for num_workers
        num_workers = num_workers or 0

        flattened_tree = []
        for tree in self.file_trees:
            filtered_tree = self._filter_tree(tree, filter=attribute_values)
            flattened_tree.extend(self._flatten_tree(filtered_tree))

        self.dataset = ResponsesDataset(
            flattened_tree,
            columns=self.columns,
        )

        loader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )

        sample = self.dataset[0]
        data = {k: [] for k in sample.keys()}
        for batch in loader:
            for k, v in batch.items():
                (
                    data[k].append(v)
                    if isinstance(v[0], torch.Tensor)
                    else data[k].extend(v)
                )

        for k, v in data.items():
            if isinstance(v[0], torch.Tensor):
                data[k] = torch.cat(v).to(torch.float32).numpy()
            else:
                data[k] = np.asarray(v)
        return data

    def as_dataframe(
        self,
        data: dict,
        columns: t.List[str] = [
            "id",
            "responses",
            "unit",
            "pooling_op",
            "module_name",
            "subset",
        ],
        explode: t.List[str] = ["responses", "unit"],
    ) -> pd.DataFrame:
        """
        Convert the loaded responses into a pandas DataFrame for further analysis and manipulation.

        Args:
            data (dict): The dictionary containing loaded responses to be converted into a DataFrame.

            columns (list, optional): Optional list of column names for the resulting DataFrame. Defaults to
                                        ["id", "responses", "unit", "pooling_op", "module_name", "subset"].

            explode (list, optional): List of columns in 'data' dictionary to be exploded into individual rows if not None.
                                        Default is ["responses", "unit"].

        Returns:
            pd.DataFrame: A pandas DataFrame containing loaded responses, with columns specified by 'columns' and indexed by 'id'.
                        If 'explode' is provided, also includes exploded rows for each unique combination of the mentioned attributes.

        Note:
            The function first converts the input dictionary into a DataFrame using provided column names. It then explodes lists
            in "responses" and "unit" columns to create multiple rows for each unique combination of these attributes, effectively
            unnesting them from their original nested structure if 'explode' is not None. Finally, it sets 'id' as the index for better data accessibility.

        Note:
            This function assumes that all values in 'data' dictionary correspond to the same 'subset', which is assumed to be common
            among all responses. If subsets differ across different response batches, they will not be consolidated by this operation.
        """
        data = pd.DataFrame(data, columns=columns).set_index("id")
        if explode is not None:
            data = data.explode(explode, ignore_index=False)
        return data

    @staticmethod
    def label_src_dst_subsets(
        data: t.Dict[str, np.ndarray],
        src_subset: t.List[str],
        dst_subset: t.List[str],
        key: str = "subset",
        balanced: bool = False,
        seed: int = 0,
    ) -> t.Dict[str, np.ndarray]:
        """
        Splits the data dictionary into source and target subsets based on specified criteria.

        This function takes a dictionary of numpy arrays `data` and two lists of subset names `src_subset` and `dst_subset`.
        It uses the data under the key `key` to determine which indices belong to each subset, then randomly shuffles these indices for balanced sampling if desired.
        The function returns two dictionaries: one containing the source subset data and another containing the target subset data.

        NOTE: The label for the source subset is 1 and 0 for the target subset.

        Parameters:
            data (dict): A dictionary where keys are strings and values are numpy arrays. The key used to access the subsets is specified by `key`.
            src_subset (list of str): List of names for the source subsets.
            dst_subset (list of str): List of names for the target subsets.
            key (str, optional): Key in `data` dictionary that holds the subset information. Default is "subset".
            balanced (bool, optional): If True, ensures each subset appears an equal number of times across both source and target dictionaries. If False, splits each subset based on its frequency within it.
            seed (int, optional): Seed for random number generation to ensure reproducibility.

        Returns:
            tuple: A tuple containing two dictionaries - one with the source subset data and another with the target subset data. Each dictionary contains key-value pairs from the input `data` corresponding to their respective subsets.

        Raises:
            AssertionError: If the set of values under the specified `key` in `data` does not match the combined list of `src_subset` and `dst_subset`.

        Example:
            data = {
                "subset": np.array(["A", "B", "C", "D"]),
                "features": np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            }
            src_subset = ["A", "B"]
            dst_subset = ["C", "D"]
            label_src_dst_subsets(data, src_subset, dst_subset)
        """
        data_subset = data[key]
        all_subsets = set(src_subset + dst_subset)
        data_subset = np.asarray(data_subset)
        all_indices = np.arange(len(data_subset))
        rng = np.random.RandomState(seed)
        subset_indices = {}
        # Count whether subset appears as only source, only target, or both
        subset_counts = defaultdict(lambda: 0)
        # Count how much data has been used in a subset in order to use the rest for the other
        subset_offsets = defaultdict(lambda: 0)
        min_subset = np.inf
        for subset in all_subsets:
            subset_indices[subset] = rng.permutation(all_indices[data_subset == subset])
            if subset in src_subset:
                subset_counts[subset] += 1
            if subset in dst_subset:
                subset_counts[subset] += 1
            min_subset = min(
                len(subset_indices[subset]) // subset_counts[subset], min_subset
            )

        ret = defaultdict(list)

        for subset in all_subsets:
            if subset in src_subset:
                if balanced:
                    end = min_subset // len(src_subset)
                    for k, v in data.items():
                        ret[k].append(v[subset_indices[subset][:end]])
                    ret["label"].append(np.ones((end,)))
                    subset_offsets[subset] += end
                else:
                    subset_len = len(subset_indices[subset]) // subset_counts[subset]
                    for k, v in data.items():
                        ret[k].append(v[subset_indices[subset][:subset_len]])
                    ret["label"].append(np.ones((subset_len,)))
                    subset_offsets[subset] += subset_len

            if subset in dst_subset:
                start = subset_offsets[subset]
                if balanced:
                    end = start + min_subset // len(dst_subset)
                    ret["label"].append(np.zeros((end - start,)))
                    for k, v in data.items():
                        ret[k].append(v[subset_indices[subset][start:end]])
                else:
                    subset_len = len(subset_indices[subset]) // subset_counts[subset]
                    for k, v in data.items():
                        ret[k].append(v[subset_indices[subset][start:]])
                    ret["label"].append(np.zeros((len(ret[k][-1]),)))
        indices = rng.permutation(sum(map(len, ret["label"])))
        ret = {k: np.concatenate(v)[indices] for k, v in ret.items()}
        return ret


class ResponsesDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that loads responses data from a list of dictionaries
    containing metadata and file paths. The data is loaded using torch.load,
    converted to numpy array for 'responses' field, and may include an additional
    'unit' number along with the response data. It also allows selection of specific
    unit for processing if needed.

    Attributes:
        records (list): A list of dictionaries containing metadata about each record
                        including file path to load data from.
        add_unit_number (bool): If True, adds 'unit' number to the loaded data.
        select_unit (int or None): If not None, only loads responses for this unit
                                    number if it exists in the data. Defaults to None.
    """

    def __init__(
        self,
        records: t.List[dict],
        add_unit_number: bool = False,
        select_unit: int = None,
        columns: t.List[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.records = records
        self.add_unit_number = add_unit_number
        self.select_unit = select_unit
        self.columns = columns
        self.dtype = dtype

    def __getitem__(self, item) -> dict:
        """
        Get a data record at the specified index.

        Args:
            item (int): The index of the record to retrieve.

        Returns:
            dict: A dictionary representing the data record, with keys and values for "responses" and "unit".
                  If `add_unit_number` is True, then "unit" will be an array of unit numbers corresponding to each response in "responses".
        """
        meta = self.records[item]
        path = meta["path"]
        datum = torch.load(path)
        datum["responses"] = datum["responses"].to(self.dtype)
        if self.add_unit_number:
            datum["unit"] = np.arange(datum["responses"].shape[-1])
            if self.select_unit is not None:
                datum["unit"] = datum["unit"][self.select_unit]
        if self.select_unit is not None:
            datum["responses"] = datum["responses"][..., self.select_unit]
        datum.update(meta)
        del datum["path"]
        if self.columns:
            datum = {k: datum[k] for k in self.columns}
        return datum

    def __len__(self) -> int:
        """
        Get the total number of data records in this dataset.

        Returns:
            int: The length (total number of data records) in this dataset.
        """
        return len(self.records)


def subsample_responses(
    responses: dict,
    size: int,
    stratified: bool = True,
    seed: int = 42,
    label_column: str = "label",
) -> dict:
    logger.info("Subsampling responses...")
    rng = np.random.RandomState(seed)
    if stratified:
        indices = np.arange(responses["responses"].shape[0])
        labels = responses[label_column]
        unique_labels = np.unique(labels)
        assert (size % len(unique_labels)) == 0
        ret = defaultdict(list)
        for label in unique_labels:
            label_indices = indices[labels == label]
            label_indices = rng.permutation(label_indices)[
                : (size // len(unique_labels))
            ]
            ret["responses"].append(responses["responses"][label_indices])
            ret[label_column].append(np.asarray(responses[label_column])[label_indices])
            ret["id"].append(np.asarray(responses["id"])[label_indices])
        ret = {k: np.concatenate(v, 0) for k, v in ret.items()}
    else:
        indices = rng.permutation(responses["responses"].shape[0])[:size]
        ret = {k: v[indices] for k, v in ret.items()}
    assert ret["responses"].shape[0] == size
    assert ret["responses"].shape[1] == responses["responses"].shape[1]
    logger.info("Done subsampling.")
    return ret
