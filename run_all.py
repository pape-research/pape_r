import runpy
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


def run_jupyter_notebook(nbpath: str):
    """Runs and saves a notebook."""

    print(f"Running notebook: {nbpath}.")

    with open(nbpath, 'r') as fh:
        nbk = nbformat.read(fh, nbformat.NO_CONVERT)

    ep = ExecutePreprocessor(timeout=None, kernel_name='python3')
    out = ep.preprocess(nbk)

    with open(nbpath, mode='w', encoding='utf-8') as fh:
        nbformat.write(nb=nbk, fp=fh)

# Comment out steps you don't want to run.

print("Starting run.")

print("Creating datasets.")
runpy.run_path(path_name='create.py')

print("Produce results.")
run_jupyter_notebook("1_run_main_experiments.ipynb")
run_jupyter_notebook("2_calculate_main_experiments_standard_error.ipynb")
run_jupyter_notebook("3_run_sample_size_effect_experiment.ipynb")
run_jupyter_notebook("4_calculate_sample_size_effect_standard_errors.ipynb")

# uncomment to run example notebook
# run_jupyter_notebook("example_performance_estimation_and_evaluation.ipynb")

print("Evaluate results.")
run_jupyter_notebook("results_01_main_results.ipynb")
run_jupyter_notebook("results_02_sample_size_effect_results.ipynb")

print("Run completed.")