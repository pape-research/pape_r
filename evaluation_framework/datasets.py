import json
import os
from dataclasses import dataclass
from typing import List, Dict
import warnings

import pandas as pd
from pandas import DataFrame



@dataclass
class DataSet:
    data_path: str
    metadata_path: str
    filename: str
    # data: pd.DataFrame
    selected_features: List[str]
    categorical_features: List[str]
    observations_in_chunk: int
    step_size: int

    def get_dataset(self):
        data_path = self.data_path
        return pd.read_parquet(data_path)


def get_paths_to_data_and_metadata(data_dir: str):

    paths_to_data_and_meta = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            if path.endswith("pq"):
                data_file_path = path
                metadata_path = path.replace('.pq', '_metadata.json')
                if not os.path.exists(metadata_path):
                    warnings.warn(
                        "Found {} but metadatafile named {} does not exist.".format(data_file_path, metadata_path)
                    )
                else:
                    paths_to_data_and_meta.append((data_file_path, metadata_path))

    return paths_to_data_and_meta


def get_datasets_for_pattern(pattern: str = None, data_dir=None) -> List[DataSet]:
    results = []

    paths_to_data_and_meta = get_paths_to_data_and_metadata(data_dir)

    for data_path, metadata_path in paths_to_data_and_meta:
        if pattern:
            if pattern not in data_path:
                continue
        data_filename = os.path.basename(data_path)
        with open(metadata_path) as infile:
            metadata = json.load(infile)
        features_selected = metadata["features_selected"]

        if 'features_categorical' in metadata:
            features_categorical = metadata['features_categorical']
        else:
            features_categorical = []

        if 'suggested_chunk_size' in metadata:
            observations_in_chunk = metadata["suggested_chunk_size"]
        else:
            observations_in_chunk = None

        if 'suggested_step_size' in metadata:
            step_size = metadata['suggested_step_size']
        else:
            step_size = None

        results.append(DataSet(
            data_path=data_path,
            metadata_path=metadata_path,
            filename=data_filename,
            selected_features=features_selected,
            categorical_features=features_categorical,
            observations_in_chunk=observations_in_chunk,
            step_size=step_size
        ))

    return results


