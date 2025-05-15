import traceback
from copy import copy

from .metrics import MetricsLogger
from .steps import *


class EvaluationPipelineMetadata(dict):
    def __init__(self,
                 experiment_name: str,
                 tags: Dict[str, str] = None,
                 params: Dict[str, Any] = None,
                 model=None,
                 model_name: str = None,
                 model_source: str = "sklearn"):
        if tags is None:
            tags = {}

        if params is None:
            params = {}

        super().__init__({
            'experiment_name': experiment_name,
            'tags': tags,
            'params': params,
            'monit_model': {
                'model': model,
                'name': model_name,
                'source': model_source,
            }
        })


def _get_model_name(model_name: Optional[str], model) -> str:
    return model_name if model_name else type(model).__name__


class EvaluationPipelineState:
    def __init__(self):
        self._state = Dict[str, Any]

    def to_dict(self):
        return self._state


class EvaluationPipeline:
    def __init__(self,
                 steps: List[Step],
                 metadata: EvaluationPipelineMetadata = None,
                 metrics_logger: MetricsLogger = None):

        self.steps = steps if steps else []
        self.data = {}
        self.metadata = None
        self.metrics_logger = None

        if metadata:
            self.metadata = metadata



    def add_custom_processing_step(self, step: Step, index: int = None):
        if not index:
            index = len(self.steps)

        processing_steps = copy(self.steps)
        processing_steps.insert(index, step)
        self.steps = processing_steps
        return self

    def reset(self):
        self.data = {}
        return self

    def set_metadata(self, metadata: EvaluationPipelineMetadata):
        self.data["metadata"] = metadata
        return self

    def run(self, df: pd.DataFrame,
            features_selected: List[str],
            features_categorical: List[str],
            observations_in_chunk: int,
            step_size: int,
            dataset_name: str = None,
            log_metrics: bool = True,
            print_steps: bool = True,
            ) -> Dict:
        """
        Runs process single dataset provided.
        """

        self.data["df_raw"] = df.copy()
        self.data["features_selected"] = copy(features_selected)
        self.data["features_categorical"] = copy(features_categorical)
        self.data["observations_in_chunk"] = copy(observations_in_chunk)
        self.data["step_size"] = copy(step_size)
        self.data["dataset_name"] = copy(dataset_name)

        if dataset_name:
            self.data['dataset'] = dataset_name

        if self.metadata:
            self.set_metadata(self.metadata)

        if log_metrics:
            if not self.metadata:
                raise RuntimeError(f'cannot log metrics without pipeline metadata')
            self.metrics_logger.start_logging(data=self.data)

        try:
            # Run user pipeline'
            for index, step in enumerate(self.steps):
                if print_steps:
                    print(
                        "Running step: {} - {}".format(
                            index, step.description
                        )
                    )

                self.data = step.run(data=self.data)
        except StepException as e:
            self.data['traceback'] = traceback.format_exc()
            print(e)
        finally:
            if log_metrics and self.metrics_logger:
                self.metrics_logger.end_logging(data=self.data)

        return self.data


class DefaultEvaluationPipeline(EvaluationPipeline):
    def __init__(self,
                 experiment_name: str,
                 observations_in_chunk: int,
                 step_size: int,
                 target_func: Callable,
                 monit_model,
                 model_name: str = None,
                 tags: Dict[str, str] = None,
                 ):
        steps = [
            SplitDataStep(),
            CombineProcessedReferenceProductionWithRawStep(),
            SplitIntoChunksStep(observations_in_chunk, step_size),
            CreatemonitFrameStep(),
            CalculateTargetOnChunksStep(target_func),
            # CalculateAggregatesOnChunksStep([np.mean, np.std], ['mean', 'std'], apply_on_preprocessed=True),
            CalculateAggregatesOnChunksStep([np.mean, np.std], ['mean', 'std'], cols_to_apply=['y_pred_proba']),
            GetmonitXyReferenceProductionStep(),
            FitmonitModelAndPredictStep(model=monit_model),
            EvaluateModelStep(evaluation_kind="regression",
                              observations_in_chunk=observations_in_chunk,
                              step_size=step_size)
        ]

        if tags is None:
            tags = {}

        metadata = EvaluationPipelineMetadata(
            experiment_name=experiment_name,
            tags=tags,
            params={
                'observations_in_chunk': observations_in_chunk,
                'step_size': step_size
            },
            model=monit_model,
            model_name=_get_model_name(model_name=model_name, model=monit_model)
        )
        super(DefaultEvaluationPipeline, self).__init__(steps=steps, metadata=metadata)
