import os.path
from typing import Dict, Callable, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .metrics import f1, roc_auc, mape, rmsle, roc_curve_custom, recall, precision


def _get_max_deviation(series):
    deviation = np.std(series) * 2
    return deviation


def _get_dictionary_with_estimators_results(
        estimators, evaluation_metrics, y_true_reference, y_true_production
):
    """
    Iterate over provided estimators and calculate each of evaluation metrics on reference and production sets.
    """
    results = {}
    for subset in ["REFERENCE", "production"]:
        for metric_name, metric in evaluation_metrics:
            for estimator_name, estimator_values in estimators.items():
                if subset == "REFERENCE":
                    score = metric(y_true_reference, estimator_values[: len(y_true_reference)])
                    results[
                        " ".join(["REFERENCE", metric_name, estimator_name])
                    ] = score

                else:
                    score = metric(y_true_production, estimator_values[len(y_true_reference):])
                    results[
                        " ".join(["production", metric_name, estimator_name])
                    ] = score

    results = {key: (np.round(item, 3) if item else np.nan) for key, item in results.items()}

    return results


def _check_data_entries(data: Dict, entries_to_check: List[str]):
    """
    Checks whether key listed in entries_to_check are available. Raises Exception if some are missing.
    """
    missing = [x for x in entries_to_check if x not in data.keys()]
    if len(missing) > 0:
        raise EvaluationException(
            "Your pipeline does not produce {}. Cannot evaluate. ".format(
                ", ".join(missing)
            )
        )


class EvaluationException(BaseException):
    pass


def evaluate_classification(data: Dict,
                            drawstyle: str = "default",
                            show_legend: bool = False,
                            classification_binary_evaluation_metrics: Dict[str, Callable] = None,
                            classification_score_evaluation_metrics: Dict[str, Callable] = None
                            ) -> Dict:
    if classification_binary_evaluation_metrics is None:
        classification_binary_evaluation_metrics = {"F1": f1, "Precision": precision, "Recall": recall}

    if classification_score_evaluation_metrics is None:
        classification_score_evaluation_metrics = {"ROC_AUC": roc_auc}

    observations_in_chunk = data["observations_in_chunk"]
    step_size = data["step_size"]

    _check_data_entries(data, ["monit_pred_reference_score", "monit_pred_production_score"])

    data = _calculate_binary_target_for_classification(data)
    data = _calculate_naive_predictions_classification(data, observations_in_chunk, step_size)
    data = _evaluate_baselines_and_prediction_classification(data,
                                                             classification_score_evaluation_metrics,
                                                             classification_binary_evaluation_metrics)
    _plot_results_classification(data, drawstyle, show_legend)

    return data


def _calculate_binary_target_for_classification(data: Dict) -> Dict:
    """
    Calculates binary target for classification task. The target is positive when the
    client model performance is n (in the first version n=2) standard deviations away from the mean on reference
    chunks.
    """

    monit_y_reference, monit_y_production = data["monit_y_reference"], data["monit_y_production"]
    deviation = _get_max_deviation(monit_y_reference)
    data["clf_max_deviation"] = deviation
    monit_y_reference_abs_diff = abs(monit_y_reference - data["score_function_on_reference"])
    monit_y_production_abs_diff = abs(monit_y_production - data["score_function_on_reference"])
    monit_y_reference_binary = np.where(monit_y_reference_abs_diff > deviation, 1, 0)
    monit_y_production_binary = np.where(monit_y_production_abs_diff > deviation, 1, 0)
    data["monit_y_reference_binary"] = monit_y_reference_binary
    data["monit_y_production_binary"] = monit_y_production_binary
    return data


def _calculate_naive_predictions_classification(data: Dict, observations_in_chunk: int, step_size: int) -> Dict:
    """
    This is functions creates a dictionary with the actual predictions and baselines.
    Creates score-based and binary baselines and predictions for classification.
    """

    (
        monit_df,
        monit_y_reference,
        monit_y_production,
        monit_pred_production_score,
        monit_pred_reference_score,
    ) = (
        data["monit_df"],
        data["monit_y_reference"],
        data["monit_y_production"],
        data["monit_pred_production_score"],
        data["monit_pred_reference_score"],
    )

    monit_pred_whole_score = pd.concat(
        [pd.Series(monit_pred_reference_score), pd.Series(monit_pred_production_score)],
        ignore_index=True,
    ).reset_index(drop=True)
    # score like estimators
    estimators_score = {"monit Model prediction": monit_pred_whole_score}
    steps_in_chunk = observations_in_chunk // step_size
    for win in [3]:  # can be modified if needed
        estimators_score[
            "BENCHMARK Persistent rolling mean window " + str(win)
            ] = abs(
            data["score_function_on_reference"]
            - monit_df["monit_y_true"].shift(steps_in_chunk).rolling(win).mean()
        )

    estimators_score["BENCHMARK Client Model Performance from reference"] = (
        pd.Series(0).repeat(len(monit_df)).reset_index(drop=True)
    )

    data["estimators_score"] = estimators_score

    # binary estimators
    if not (
            "monit_pred_reference_binary" in data.keys()
            and "monit_pred_production_binary" in data.keys()
    ):
        data["monit_pred_reference_binary"] = pd.Series(np.nan).repeat(
            data["n_reference_chunks"]
        )
        data["monit_pred_production_binary"] = pd.Series(np.nan).repeat(
            data["n_production_chunks"]
        )

    max_deviation = data["clf_max_deviation"]
    estimators_binary = {}
    for estimator_name, estimator_data in estimators_score.items():
        if estimator_name == "monit Model prediction":
            monit_pred_whole = pd.concat(
                [
                    pd.Series(data["monit_pred_reference_binary"]),
                    pd.Series(data["monit_pred_production_binary"]),
                ],
                ignore_index=True,
            ).reset_index(drop=True)
            estimators_binary[estimator_name] = monit_pred_whole
        else:
            estimators_binary[estimator_name] = np.where(
                estimator_data > max_deviation, 1, 0
            )

    data["estimators_binary"] = estimators_binary

    return data


def _evaluate_baselines_and_prediction_classification(data: Dict,
                                                      classification_score_evaluation_metrics: Dict[str, Callable],
                                                      classification_binary_evaluation_metrics: Dict[str, Callable]
                                                      ) -> Dict:
    """
    Calculate scores on reference and production for all classification estimators.
    """

    results_scores = _get_dictionary_with_estimators_results(
        data["estimators_score"],
        classification_score_evaluation_metrics.items(),
        data["monit_y_reference_binary"],
        data["monit_y_production_binary"],
    )

    results_binary = _get_dictionary_with_estimators_results(
        data["estimators_binary"],
        classification_binary_evaluation_metrics.items(),
        data["monit_y_reference_binary"],
        data["monit_y_production_binary"],
    )

    results = results_scores
    results.update(results_binary)
    results = {key: results[key] for key in sorted(results)[::-1]}

    data["results"] = results

    # calculate fpr, tpr and roc_curve
    if "results_vectors" not in data.keys():
        data["results_vectors"] = {}

    estimators_score = data["estimators_score"]
    for estimator_name, estimator_data in estimators_score.items():
        fpr, tpr, thresholds = roc_curve_custom(
            data["monit_y_production_binary"], estimator_data.iloc[-data["n_production_chunks"]:]
        )

        data['results_vectors'][estimator_name] = \
            {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds
            }

    return data


def _plot_results_classification(data: Dict, drawstyle, show_legend):
    """
    Plot the results for classification.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    if 'dataset' in data.keys():
        fig.suptitle(data['dataset'])
    if "results_vectors" in data.keys():
        for estimator_name, result_vectors in data['results_vectors'].items():
            fpr, tpr = result_vectors["fpr"], result_vectors["tpr"]
            if (fpr is not None) and (tpr is not None):
                if estimator_name == "monit Model prediction":  # change style for actual prediction
                    ax1.plot(fpr, tpr, label=estimator_name, drawstyle=drawstyle)
                else:
                    ax1.plot(
                        fpr,
                        tpr,
                        label=estimator_name,
                        alpha=0.7,
                        linestyle="--",
                        drawstyle=drawstyle,
                    )
        ax2.set_xlabel("fpr")
        ax2.set_ylabel("tpr")
        if show_legend:
            ax1.legend()

    upper_limit = data["score_function_on_reference"] + data["clf_max_deviation"]
    lower_limit = data["score_function_on_reference"] - data["clf_max_deviation"]
    ax2.set_xlabel("Client model performance")
    ax2.set_ylabel("monit model score prediction")
    ax2.scatter(data["monit_y_production"], data["monit_pred_production_score"], label='production', alpha=0.5, color='red')
    ax2.scatter(data["monit_y_reference"], data["monit_pred_reference_score"], label='reference', alpha=0.5, color='grey')
    ax2.axvline(x=lower_limit, linestyle="--", alpha=0.8, color='black', label='2std boundary')
    ax2.axvline(x=upper_limit, linestyle="--", alpha=0.8, color='black')
    if show_legend:
        ax2.legend()
    plt.show()


def evaluate_regression(data: Dict,
                        drawstyle: str = "default",
                        show_legend: bool = False,
                        regression_evaluation_metrics: Dict[str, Callable] = None,
                        ) -> Dict:
    if not regression_evaluation_metrics:
        regression_evaluation_metrics = {"MAPE": mape, "RMSLE": rmsle}

    observations_in_chunk = data["observations_in_chunk"]
    step_size = data["step_size"]

    _check_data_entries(data, ["monit_pred_reference", "monit_pred_production"])

    monit_pred_reference, monit_pred_production = (
        data["monit_pred_reference"],
        data["monit_pred_production"],
    )
    monit_pred_whole = pd.concat(
        [pd.Series(monit_pred_reference), pd.Series(monit_pred_production)],
        ignore_index=True,
    )
    data["monit_pred_whole"] = monit_pred_whole

    _calculate_naive_predictions_regression(data, observations_in_chunk, step_size)
    _evaluate_baselines_and_prediction_regression(data, regression_evaluation_metrics)
    _plot_results_regression(data, drawstyle, show_legend)

    return data


def _calculate_naive_predictions_regression(data: Dict, observations_in_chunk: int, step_size: int) -> Dict:
    """
    This is functions creates a dictionary with the actual predictions and baselines.
    For now baselines are:
    - conservative persistent prediction i.e. the mean of the target from the whole reference set
    - rolling persistent prediction:
        - mean of target from 3 previous chunks (if step_between_chunks < observations_in_chunk
         then the mean is calculated from chunks that are k chunks before
         (where k = observations_in_chunk//step_between chunks to ensure that chunks used for calculating evaluation
         do not contain observations from the chunk evaluated)
    Creates 'estimators' entry in the dictionary that contains vectors with predictions.
    """

    # mean from previous
    monit_df, monit_y_reference, monit_y_production, monit_pred_production, monit_pred_reference = (
        data["monit_df"],
        data["monit_y_reference"],
        data["monit_y_production"],
        data["monit_pred_production"],
        data["monit_pred_reference"],
    )

    monit_pred_whole = pd.concat(
        [pd.Series(monit_pred_reference), pd.Series(monit_pred_production)], ignore_index=True
    ).reset_index(drop=True)

    estimators = {"monit Model prediction": monit_pred_whole}

    steps_in_chunk = observations_in_chunk // step_size
    for win in [3]:  # can be modified if needed
        estimators["BENCHMARK Persistent rolling mean window " + str(win)] = (
            monit_df["monit_y_true"].shift(steps_in_chunk).rolling(win).mean()
        )

    estimators["BENCHMARK Client Model Performance from reference"] = (
        pd.Series(data["score_function_on_reference"])
        .repeat(len(monit_df))
        .reset_index(drop=True)
    )

    data["estimators"] = estimators

    return data


def _evaluate_baselines_and_prediction_regression(data: Dict,
                                                  regression_evaluation_metrics: Dict[str, Callable]) -> Dict:
    """
    Calculate scores on reference and production for all regression estimators.
    """

    monit_y_reference, monit_y_production, monit_pred_production = (
        data["monit_y_reference"],
        data["monit_y_production"],
        data["monit_pred_production"],
    )
    estimators = data["estimators"]
    evaluation_metrics = regression_evaluation_metrics.items()

    results = _get_dictionary_with_estimators_results(
        estimators, evaluation_metrics, monit_y_reference, monit_y_production
    )

    # custom Mean abs relative err = (y_true - monit_y_pred-)/(y_true - persistent_baseline_pred)
    relative_distance = abs(
        (
            monit_y_production
            - estimators["BENCHMARK Client Model Performance from reference"][
                len(monit_y_reference):
            ]
        )
    ) - abs((monit_y_production - monit_pred_production))

    data["results"], data["relative_distance"] = results, relative_distance

    return data


# TODO: remove duplication with categorical plot
def _plot_results_regression(data: Dict, drawstyle: str, show_legend: bool):
    """
    Plots the results for regression.
    """
    monit_df, monit_y_reference = data["monit_df"], data["monit_y_reference"]
    estimators, relative_distance = data["estimators"], data["relative_distance"]

    # calculate std for alert line
    std_on_reference = np.std(monit_y_reference)
    upper_limit = data["score_function_on_reference"] + 2 * std_on_reference
    lower_limit = data["score_function_on_reference"] - 2 * std_on_reference

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    if 'dataset' in data.keys():
        fig.suptitle(data['dataset'])

    ax1.plot(
        monit_df.index,
        monit_df["monit_y_true"],
        label="monit target",
        drawstyle=drawstyle,
    )

    for estimator_name, estimator_data in estimators.items():
        if (
                estimator_name == "monit Model prediction"
        ):  # change style for actual prediction
            ax1.plot(estimator_data, label=estimator_name, drawstyle=drawstyle)
        else:
            ax1.plot(
                estimator_data,
                label=estimator_name,
                alpha=0.7,
                linestyle="--",
                drawstyle=drawstyle,
            )

    ax1.axvline(
        x=data['last_reference_chunk'],
        label="Last monit reference chunk",
        color="black",
        linestyle="--",
        alpha=0.8,
    )
    ax1.axvline(
        x=data['first_production_chunk'],
        label="First clear production chunk",
        color="red",
        linestyle="--",
        alpha=0.8,
    )
    ax1.axhline(
        y=upper_limit,
        label="Upper and lower - production score +/- 2std",
        color="brown",
        linestyle="--",
        alpha=0.8,
    )
    ax1.axhline(y=lower_limit, color="brown", linestyle="--", alpha=0.8)
    ax1.set_ylabel("performance metric")
    ax1.set_xlabel("chunk")
    if show_legend:
        ax1.legend()
    ax2.scatter(data['monit_y_production'], data['monit_pred_production'], label="production data")
    max_plot_val = np.maximum(np.max(data['monit_y_production']), np.max(data['monit_pred_production']))
    min_plot_val = np.minimum(np.min(data['monit_y_production']), np.min(data['monit_pred_production']))
    ax2.plot([min_plot_val, max_plot_val], [min_plot_val, max_plot_val], color='red', linestyle=':', label='y=x')

    ax2.set_ylabel("monit prediction")
    ax2.set_xlabel("y true")
    ax2.axvline(x=lower_limit, linestyle="--", alpha=0.8, color='black', label='2std boundary')
    ax2.axvline(x=upper_limit, linestyle="--", alpha=0.8, color='black')
    ax2.axhline(y=lower_limit, linestyle="--", alpha=0.8, color='black')
    ax2.axhline(y=upper_limit, linestyle="--", alpha=0.8, color='black')
    if show_legend:
        ax2.legend()
    plt.show()




def evaluate_comparison_single(data: Dict, drawstyle: str = "default", show_legend: bool = False, compare_data_entries_names=[], plot=True, scatter_entries=[], save_fig=False,
                 fig_location=None, fig_name:str = '') -> Dict:

    observations_in_chunk = data["observations_in_chunk"]
    step_size = data["step_size"]

    _check_data_entries(data, compare_data_entries_names)

    len_reference= data["n_reference_chunks"]

    for entry in compare_data_entries_names:
        data[entry + '_reference'] = data[entry][:len_reference]
        data[entry + '_production'] = data[entry][len_reference:]

    if plot:
        if 'dataset_name' in data.keys():
            dataset_name = data['dataset_name']
        else:
            dataset_name = 'Dataset_name_not_specified'

        if not fig_location:
            fig_location = "EF_plots"
        print("plotting")
        _plot_results_comparison(data, drawstyle, show_legend, compare_data_entries_names, scatter_entries, save_fig, fig_location, dataset_name, fig_name)

    return data

def _plot_results_comparison(data: Dict, drawstyle: str, show_legend: bool, compare_data_entries_names: List, scatter_entries: List,
                             save_fig: bool, fig_location: str, dataset_name: str, fig_name: str):
    """
    Plots the results for regression.
    """
    # monit_df, monit_pred_reference = data["monit_df"], data["monit_pred_reference"]
    #
    # # calculate std for alert line
    # std_on_reference = np.std(monit_pred_reference)
    # upper_limit = np.mean(monit_pred_reference) + 2 * std_on_reference
    # lower_limit = np.mean(monit_pred_reference) - 2 * std_on_reference

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    if 'dataset' in data.keys():
        fig.suptitle(data['dataset'])

    ax1_2 = ax1.twinx()

    for i, entry in enumerate(compare_data_entries_names):
        if i==0:
            ax1.plot(
                range(len(data[entry])),
                data[entry],
                label="Benchmark - " + entry,
                alpha=0.7,
                color='black',
                drawstyle=drawstyle,
            )
        else:
            ax1_2.plot(
                range(len(data[entry])),
                data[entry],
                label=entry,
                drawstyle=drawstyle,
            )


    ax1.axvline(
        x=data['last_reference_chunk'],
        label="Last monit reference chunk",
        color="black",
        linestyle="--",
        alpha=0.8,
    )
    ax1.axvline(
        x=data['first_production_chunk'],
        label="First clear production chunk",
        color="red",
        linestyle="--",
        alpha=0.8,
    )


    ax1.set_ylabel("Benchmark")
    ax1_2.set_ylabel("Evaluated method/s")
    ax1.set_xlabel("chunk")
    if show_legend:
        ax1.legend()



    for i, entry in enumerate(scatter_entries[1:]):
        ax2.scatter(data[scatter_entries[0]], data[entry], label=entry)

    ax2.set_title('Scatter plot on production data')
    ax2.set_xlabel(scatter_entries[0])

    if show_legend:
        ax2.legend()

    if save_fig:
        if not os.path.exists(fig_location):
            os.makedirs(fig_location)

        plt.savefig(os.path.join(fig_location, dataset_name + '_' + fig_name + '.png'), format='png', dpi=250)

    plt.show()


def evaluate_comparison_many(data: Dict, drawstyle: str = "default", show_legend: bool = False, compare_data_entries_suffix=[]) -> Dict:

    observations_in_chunk = data["observations_in_chunk"]
    step_size = data["step_size"]
    features_selected = data['features_selected']

    all_entries = []
    for feature in features_selected:
        all_entries += [feature + x for x in compare_data_entries_suffix]

    _check_data_entries(data, all_entries)

    len_reference= data["n_reference_chunks"]

    for entry in all_entries:
        data[entry + '_reference'] = data[entry][:len_reference]
        data[entry + '_production'] = data[entry][len_reference:]

    for feature in features_selected:
        entries_related_to_feature = [feature + x for x in compare_data_entries_suffix]
        print("Comparison related to feature {}".format(feature))
        _plot_results_comparison(data, drawstyle, show_legend, entries_related_to_feature)

    return data