from typing import Callable, Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from .evaluation import evaluate_regression, evaluate_classification, evaluate_comparison_single, evaluate_comparison_many


class Step:
    def __init__(self, description: str, func: Callable[[Dict, Optional[Any]], Dict], args: Tuple = None,
                 conditional_func: Callable[[Dict], bool] = None):
        self.description = description
        self.func = func
        self.args = args
        self.conditional_func = conditional_func

    def run(self, data: Dict) -> Dict:
        try:
            if not self.conditional_func or self.conditional_func(data):
                if self.args:
                    return self.func(data, *self.args)
                else:
                    return self.func(data)
            else:
                return data
        except Exception as exc:
            raise StepException(f'could not execute step {self.description}: {exc}')


class StepException(BaseException):
    pass


def _split_raw_to_reference_production(data: Dict) -> Dict:
    df = data["df_raw"]
    df_train = df[df["partition"] == "train"].copy()
    df_reference = df[df["partition"] == "reference"].copy()
    df_production = df[df["partition"] == "production"].copy()
    data["df_reference"], data["df_production"], data["df_train"] = df_reference, df_production, df_train
    return data


class SplitDataStep(Step):
    """
    Splits raw data to reference and production subsets based on 'partition' columns.

    Creates 'df_reference' and 'df_production' entries in data dictionary.
    """

    def __init__(self):
        super(SplitDataStep, self).__init__(
            description="Split raw data to reference and production",
            args=None,
            func=_split_raw_to_reference_production
        )


def _fit_on_reference_transform_all(data: Dict, pipeline, fit_on, create_snapshot, snapshot_name) -> Dict:
    if fit_on == "raw":
        df_reference, df_production, features_selected = (
            data["df_reference"],
            data["df_production"],
            data["features_selected"],
        )
    else:
        df_reference, df_production, features_selected = (
            data["preprocessed_reference"],
            data["preprocessed_production"],
            data["preprocessed_cols"],
        )

    fitted_pipeline = pipeline.fit(df_reference[features_selected])
    preprocessed_reference = pd.DataFrame(
        fitted_pipeline.transform(df_reference[features_selected])
    )
    preprocessed_production = pd.DataFrame(
        fitted_pipeline.transform(df_production[features_selected])
    )

    preprocessed_column_names = ["PF" + str(x) for x in preprocessed_reference.columns]
    preprocessed_reference.columns = preprocessed_column_names
    preprocessed_production.columns = preprocessed_column_names

    (
        data["preprocessed_reference"],
        data["preprocessed_production"],
        data["preprocessed_cols"],
    ) = (
        preprocessed_reference.copy(),
        preprocessed_production.copy(),
        preprocessed_column_names,
    )

    data_snapshot = {
        "preprocessed_reference": preprocessed_reference,
        "preprocessed_production": preprocessed_production,
        "preprocessed_cols": preprocessed_column_names,
        "pipeline_object": fitted_pipeline,
    }
    if create_snapshot:
        data[snapshot_name] = data_snapshot

    return data


class FitOnReferenceTransformAllStep(Step):
    """
    Facilitates fitting objects on reference and transforming on production. As a result, preprocessed_reference
    and preprocessed_production are created.
    args:
    - pipeline (class instance, eg. sklearn.pipeline.Pipeline) - have methods .fit and .transform
    - step_description (str) - informative description of processing step
    - fit_on (str) - useful when adding multiple separate pipelines, the first should work on
        df_reference and df_production (i.e. raw splitted data), in that case fit_on should be equal to 'raw'.
        If adding another pipeline that should work on modified data, fit_on should be != 'raw'.
        In such case preprocessed_reference and preprocessed_production will be taken as inupts.

    Creates 'preprocess_reference', 'preprocessed_production' and 'preprocessed_cols' entries in data dictionary.
    Fitted pipeline as well as snapshots of data are saved in 'data_snapshot_after_step_#'.

    """

    def __init__(self,
                 description: str,
                 pipeline,
                 fit_on: str = "raw",
                 create_snapshot: bool = True,
                 snapshot_name: str = None):
        if snapshot_name is None:
            snapshot_name = "data_snapshot_after_" + description.replace(' ', '_').lower()

        super(FitOnReferenceTransformAllStep, self).__init__(
            description=description,
            args=(pipeline, fit_on, create_snapshot, snapshot_name),
            func=_fit_on_reference_transform_all
        )


def _combine_processed_reference_and_production_data_with_raw(data: Dict) -> Dict:
    """
    This is a runner of the respective public method. See the public method docstring.
    """
    # if no preprocessing
    if "preprocessed_reference" not in data.keys():
        data["preprocessed_reference"] = data["df_reference"]
    else:
        preprocessed_reference, df_reference, = (
            data["preprocessed_reference"],
            data["df_reference"],
        )
        preprocessed_reference = pd.concat([preprocessed_reference, df_reference], axis=1)
        data["preprocessed_reference"] = preprocessed_reference

    if "preprocessed_production" not in data.keys():
        data["preprocessed_production"] = data["df_production"]
    else:
        preprocessed_production, df_production = data["preprocessed_production"], data["df_production"]
        preprocessed_production = pd.concat(
            [preprocessed_production, df_production], axis=1
        )
        data["preprocessed_production"] = preprocessed_production
    return data


class CombineProcessedReferenceProductionWithRawStep(Step):
    def __init__(self):
        super(CombineProcessedReferenceProductionWithRawStep, self).__init__(
            description="Combine processed with raw to get access to 'y_pred_proba' and 'y_true'",
            args=None,
            func=_combine_processed_reference_and_production_data_with_raw
        )


def _split_dataset_into_chunks(data: Dict) -> Dict:
    preprocessed_reference, preprocessed_production = (
        data["preprocessed_reference"],
        data["preprocessed_production"],
    )

    observations_in_chunk = data["observations_in_chunk"]
    step_size = data["step_size"]

    df = pd.concat([preprocessed_reference, preprocessed_production], axis=0).reset_index(
        drop=True
    )
    list_with_chunks = [
        df.loc[i: i + observations_in_chunk - 1, :]
        for i in range(0, len(df), step_size)
        if i + observations_in_chunk - 1 < len(df)
    ]

    reference_chunks = [
        1 if (df["partition"].iloc[-1] == "reference") else 0 for df in list_with_chunks
    ]
    production_chunks = [
        1 if (df["partition"].iloc[0] == "production") else 0 for df in list_with_chunks
    ]
    transition_chunks = [
        1
        if (
                (df["partition"].iloc[0] == "reference")
                and (df["partition"].iloc[-1] == "production")
        )
        else 0
        for df in list_with_chunks
    ]
    n_reference_chunks = np.sum(reference_chunks)
    n_transition_chunks = np.sum(transition_chunks)
    n_production_chunks = np.sum(production_chunks) + n_transition_chunks

    last_reference_chunk = np.nonzero(reference_chunks)[0][-1]
    first_production_chunk = np.nonzero(production_chunks)[0][0]

    data["last_reference_chunk"] = last_reference_chunk
    data["first_production_chunk"] = first_production_chunk
    n_chunks = len(list_with_chunks)

    print(
        "Chunk size: {}, all chunks: {}, reference_chunks: {}, "
        "transition chunks: {}, production_chunks (includes transition) :{}".format(
            len(list_with_chunks[0]),
            n_chunks,
            n_reference_chunks,
            n_transition_chunks,
            n_production_chunks,
        )
    )

    (
        data["list_with_chunks"],
        data["n_chunks"],
        data["n_reference_chunks"],
        data["n_transition_chunks"],
        data["n_production_chunks"],
    ) = (
        list_with_chunks,
        n_chunks,
        n_reference_chunks,
        n_transition_chunks,
        n_production_chunks,
    )
    return data


class SplitIntoChunksStep(Step):
    """
    Concatenates 'preprocessed_reference' with 'preprocessed_production' and then split to chunks of size
    defined on instance init.
    Creates following entries: 'list_with_chunks' - list with dataframes, 'n_chunks' - number of chunks,
    'n_reference_chunks' - number of reference chunks,  'n_production_chunks' - number of production chunks.
    """

    def __init__(self):
        super(SplitIntoChunksStep, self).__init__(
            description="Split preprocessed data into chunks that will be points for monit model",
            args=None,
            func=_split_dataset_into_chunks
        )


def _create_monit_df(data: Dict) -> Dict:
    partition = ["reference"] * data["n_reference_chunks"] + ["production"] * data[
        "n_production_chunks"
    ]
    monit_df = pd.DataFrame({"partition": partition})
    data["monit_df"] = monit_df
    return data


class CreatemonitFrameStep(Step):
    """
    Creates DataFrame with each row representing each monit point i.e. each chunk. Only 'partition'
    column is created in this step.

    Creates entry: 'monit_df'.
    """

    def __init__(self):
        super(CreatemonitFrameStep, self).__init__(
            description="Create placeholder for monit df",
            args=None,
            func=_create_monit_df
        )


def _calculate_target_on_chunks(data: Dict, target_function) -> Dict:
    """
    This is a runner of the respective public method. See the public method docstring.
    """
    monit_df, list_with_chunks = data["monit_df"], data["list_with_chunks"]

    for i, chunk in enumerate(list_with_chunks):
        monit_df.loc[i, "monit_y_true"] = target_function(chunk)

    data["score_function_on_reference"] = target_function(data["df_reference"])
    data["score_function_on_production"] = target_function(data["df_production"])
    return data


class CalculateTargetOnChunksStep(Step):
    """
    Creates 'monit_y_true' in 'monit_df' DataFrame based on the target function provided.
    Example of target function (notice that column names to use are 'y_true' and 'y_pred_proba'):

    def target_function(chunk):
        return sklearn.metrics.log_loss(chunk['y_true'], chunk['y_pred_proba'])

    """

    def __init__(self, target_func: Callable):
        super(CalculateTargetOnChunksStep, self).__init__(
            description="Calculate_target_on_chunks",
            args=(target_func,),
            func=_calculate_target_on_chunks
        )


def _calculate_aggregates_on_chunks(
        data: Dict, agg_functions: List[Callable], agg_function_names: List[str],
        cols_to_apply: List[str], apply_on_preprocessed: bool) -> Dict:
    monit_df, list_with_chunks = data["monit_df"], data["list_with_chunks"]

    if apply_on_preprocessed:
        cols = data["preprocessed_cols"]
    else:
        cols = cols_to_apply
    for col in cols:
        for function, function_name in zip(agg_functions, agg_function_names):
            for i, chunk in enumerate(list_with_chunks):
                monit_df.loc[i, col + "_" + function_name] = function(chunk[col])

    return data


class CalculateAggregatesOnChunksStep(Step):
    """
    Facilitates calculating aggregates on chunk. Takes each function from 'agg_functions' and
    applies it on each column in 'cols_to_apply' (if apply_on_processed=False, and 'cols_to_apply'
    provided) or on each 'preprocessing cols' to calculate aggregation on chunk and save in
    'monit_df' (if apply_on_preprocessed=True). For example, for:
     - agg_functions=[np.mean],
     - agg_function_names=['mean']
     - cols_to_apply=['y_pred_proba']
     - apply_on_preprocessed=False
    In 'monit_df' a column named 'y_pred_proba_mean' will be created and in each row np.mean from
    respective chunk will be calculated.
    """

    def __init__(self, agg_functions: List[Callable], agg_function_names: List[str],
                 cols_to_apply: List[str] = None, apply_on_preprocessed: bool = False):

        if cols_to_apply is None:
            cols_to_apply = []

        func_names_str = ", ".join(agg_function_names)
        if apply_on_preprocessed:
            desc = "Calculate {} on chunks, on preprocessed columns.".format(
                func_names_str
            )
        else:
            desc = "Calculate {} on chunks, on {}.".format(
                func_names_str, ", ".join(cols_to_apply)
            )

        super(CalculateAggregatesOnChunksStep, self).__init__(
            description=desc,
            args=(agg_functions, agg_function_names, cols_to_apply, apply_on_preprocessed),
            func=_calculate_aggregates_on_chunks
        )


def _get_monit_X_y_reference_production(data: Dict) -> Dict:
    """
    This is a runner of the respective public method. See the public method docstring.
    """

    monit_df = data["monit_df"]
    monit_X_reference = monit_df[monit_df["partition"] == "reference"].copy()
    monit_y_reference = monit_X_reference.pop("monit_y_true")

    monit_X_production = monit_df[monit_df["partition"] == "production"].copy()
    monit_y_production = monit_X_production.pop("monit_y_true")

    monit_X_reference.drop(columns=["partition"], inplace=True)
    monit_X_production.drop(columns=["partition"], inplace=True)

    (
        data["monit_X_reference"],
        data["monit_y_reference"],
        data["monit_X_production"],
        data["monit_y_production"],
    ) = (monit_X_reference, monit_y_reference, monit_X_production, monit_y_production)

    return data


class GetmonitXyReferenceProductionStep(Step):
    """
    Creates 'monit_X_reference', 'monit_y_reference', 'monit_X_production', 'monit_y_production' from 'monit_df' based on
    partition.
    """

    def __init__(self):
        super(GetmonitXyReferenceproductionStep, self).__init__(
            description="Split monit to X_reference, y_reference, X_production, y_production",
            args=None,
            func=_get_monit_X_y_reference_production
        )


def _fit_monit_model_and_predict(data: Dict, model) -> Dict:
    monit_X_reference, monit_y_reference, monit_X_production, monit_y_production = (
        data["monit_X_reference"],
        data["monit_y_reference"],
        data["monit_X_production"],
        data["monit_y_production"],
    )

    model.fit(monit_X_reference, monit_y_reference)
    monit_pred_reference = model.predict(monit_X_reference)
    monit_pred_production = model.predict(monit_X_production)

    data["monit_pred_reference"], data["monit_pred_production"] = (
        monit_pred_reference,
        monit_pred_production,
    )

    return data


class FitmonitModelAndPredictStep(Step):
    """
    Fits the specified algorithm ('model') on ('monit_X_reference', 'monit_y_reference') and predicts on
    'monit_X_reference' and 'monit_X_production'.
    Creates entries: 'monit_pred_reference', 'monit_pred_production', 'monit_pred_whole'.
    """

    def __init__(self, model):
        super(FitmonitModelAndPredictStep, self).__init__(
            description="Fit model and predict",
            args=(model,),
            func=_fit_monit_model_and_predict
        )


def _eval_regression(data: Dict, *args) -> Dict:
    return evaluate_regression(data, *args)


def _eval_classification(data: Dict, *args) -> Dict:
    return evaluate_classification(data, *args)

def _eval_comparison_single(data: Dict, *args) -> Dict:
    return evaluate_comparison_single(data, *args)

def _eval_comparison_many(data: Dict, *args) -> Dict:
    return evaluate_comparison_many(data, *args)


class EvaluateModelStep(Step):
    """
    Evaluates the predictions made by the model using a set of given metrics.
    Has multiple implementations based on the case of problem that is represented (e.g. regression, classification).
    Select the one you need using the `evaluation_kind` parameter.
    Creates entries: 'results', 'estimators'
    """

    def __init__(self,
                 evaluation_kind: str,
                 drawstyle: str = None,
                 show_legend: bool = False,
                 classification_score_evaluation_metrics: Dict[str, Callable] = None,
                 classification_binary_evaluation_metrics: Dict[str, Callable] = None,
                 regression_evaluation_metrics: Dict[str, Callable] = None,
                 ):

        if evaluation_kind == 'regression':
            desc = "Evaluate regression model"
            args = (drawstyle,
                    show_legend,
                    regression_evaluation_metrics)
            eval_func = _eval_regression
        elif evaluation_kind == 'classification':
            desc = "Evaluate classification model"
            args = (
                drawstyle,
                show_legend,
                classification_binary_evaluation_metrics,
                classification_score_evaluation_metrics)
            eval_func = _eval_classification
        else:
            raise StepException(f'unknown evaluation_kind: {evaluation_kind}')

        super(EvaluateModelStep, self).__init__(
            description=desc,
            args=args,
            func=eval_func
        )

class ComparePlotSingle(Step):
    """
    Enables plotting different solutions to drift detection/performance estimation on single plot per dataset.
    In compare_data_entries_names provide list of data entries names (strings) which are vectors (lists, pd.Series, np.arrays)
    of length of chunk. First entry should be the benchmark solution (the one we compare to).
    """

    def __init__(self,
                 drawstyle: str = None,
                 show_legend: bool = False,
                 compare_data_entries_names: List = [],
                 plot=True,
                 scatter_entries: List = [],
                 save_fig=False,
                 fig_location=None,
                 fig_name='',
                 ):

        if len(compare_data_entries_names) == 0:
            raise StepException(f'Provide list of data objects to compare by specyfing their names in compare_data_entries_names argument')

        desc = "Calculate comparison of different solutions."
        args = (drawstyle,
                show_legend,
                compare_data_entries_names,
                plot,
                scatter_entries,
                save_fig,
                fig_location,
                fig_name
                )
        eval_func = _eval_comparison_single

        super(ComparePlotSingle, self).__init__(
            description=desc,
            args=args,
            func=eval_func
        )


class ComparePlotPerFeature(Step):
    """
    Enables plotting different solutions to drift detection/performance estimation on many plots per dataset - one for each feature.
    In compare_data_entries_suffix provide list of data entries names (strings) which are vectors (lists, pd.Series, np.arrays) and end with suffix.
    The prefix is the feature name. First entry should be the benchmark solution (the one we compare to).
    """

    def __init__(self,
                 drawstyle: str = None,
                 show_legend: bool = False,
                 compare_data_entries_suffix: List = [],
                 ):

        if len(compare_data_entries_suffix) == 0:
            raise StepException(f'Provide list of data objects to compare by specyfing their names in compare_data_entries_names argument')

        desc = "Plot comparison of different solutions."
        args = (drawstyle,
                show_legend,
                compare_data_entries_suffix)
        eval_func = _eval_comparison_many

        super(ComparePlotPerFeature, self).__init__(
            description=desc,
            args=args,
            func=eval_func
        )


