import os
from typing import List, Tuple, Dict, Any, Optional


import pandas as pd
import traceback
from .datasets import DataSet
from .pipeline import EvaluationPipeline
from .steps import StepException


def _modify_columns_to_multindex(df, n_splits=3):
    """
    Creates multiindex columns from flat column names. nsplits determines the number of top levels. If
    nsplits=0 - nothing changes, for nsplits=1 first word of each column is used as top level index, for
    nsplits=2, first and second word are used etc.
    """
    midx_list = [
        (
                [col.split(" ")[n] for n in range(n_splits)]
                + [" ".join(col.split(" ")[n_splits:])]
        )
        for col in df.columns
    ]

    return pd.DataFrame(df.values, columns=pd.MultiIndex.from_tuples(midx_list), index=df.index)


def run_on_single_dataset(
        df: pd.DataFrame,
        features_selected: List[str],
        features_categorical: List[str],
        observations_in_chunk: int,
        step_size: int,
        pipeline: EvaluationPipeline,
        dataset_name: str = None,
        print_steps: bool = True,
        log_metrics: bool = True

) -> Tuple[Any, Dict]:
    """
    Runs an evaluation pipeline for a single given DataFrame.

    @param df: the `pandas.DataFrame` containing the model inputs/outputs.
    @param features_selected: an array containing the column names of the `pandas.DataFrame`
    to be included in the evaluation.
    @param pipeline: a `EvaluationPipeline` object that represents a custom evaluation pipeline.
    @param dataset_name: a display name for the current dataset, used in plots.
    @param print_steps: prints an overview of pipeline steps whilst running the pipeline.
    @param log_metrics: enable logging parameters, metrics and artifacts of these runs to MLFlow.

    @return: a tuple containing the evaluation results as the first element and the full pipeline state as the second.

    """
    pipeline.reset()
    data = pipeline.run(
        df=df,
        features_selected=features_selected,
        features_categorical=features_categorical,
        observations_in_chunk=observations_in_chunk,
        step_size=step_size,
        dataset_name=dataset_name,
        print_steps=print_steps,
        log_metrics=log_metrics,
    )
    results = {}
    if 'results' in data:
        results = data["results"]

    experiment_data = data

    return results, experiment_data


def run_on_many_datasets(
        datasets: List[DataSet],
        pipeline: EvaluationPipeline,
        continue_on_exception: bool = True,
        print_steps: bool = False,
        log_metrics: bool = True
):
    # print steps
    if print_steps:
        print("Steps to run:")
        for idx, step in enumerate(pipeline.steps):
            print("Step {} - {}".format(idx, step.description))

    df_with_results = pd.DataFrame()
    all_experiments_data = []
    for dataset in datasets:
        print(f'\nEvaluating on: {dataset.data_path}')
        try:
            results, experiment_data = run_on_single_dataset(
                df=dataset.get_dataset(),
                features_selected=dataset.selected_features,
                features_categorical= dataset.categorical_features,
                observations_in_chunk=dataset.observations_in_chunk,
                step_size=dataset.step_size,
                pipeline=pipeline,
                dataset_name=os.path.basename(os.path.normpath(dataset.data_path)),
                print_steps=print_steps,
                log_metrics=log_metrics,
            )
            results_row = pd.DataFrame.from_records([results], index=[dataset.filename])
            # df_with_results = df_with_results.append(results_series, ignore_index=False)
            df_with_results = pd.concat([df_with_results, results_row], ignore_index=False)
            all_experiments_data.append(experiment_data)

            pipeline.reset()  # clear all pipeline state
        except (Exception, StepException) as e:
            print("Couldn't run on {}, reason {}".format(dataset.data_path, e))
            print(traceback.format_exc())
            if not continue_on_exception:
                raise

    if not df_with_results.empty:
        df_with_results = _modify_columns_to_multindex(df_with_results)

    return df_with_results, all_experiments_data
