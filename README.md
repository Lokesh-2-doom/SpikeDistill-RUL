# SpikeDistill-RUL

Remaining useful life (RUL) prediction for NASA C-MAPSS turbofan engines using an attention-BiLSTM teacher and a recurrent leaky integrate-and-fire (LIF) spiking neural network (SNN) student.

This repository contains the project source code and a reproducible workflow notebook. Dataset files, trained checkpoints, generated metrics, and submission documents are not included.

## Project overview

The project evaluates FD001, FD002, FD003, and FD004. Sensor records are preprocessed into chronological 30-cycle windows, and RUL targets are capped at 125 cycles. Validation splits are made by engine. Preprocessing parameters are fitted on training data and applied to validation/test data.

The teacher is a bidirectional LSTM with 96 hidden units in each direction and temporal attention. The student is a recurrent LIF SNN with learned leak and firing thresholds, surrogate-gradient training, and temporal-attention readout. During development, the project also tested classical regressors, feed-forward and recurrent baselines, direct SNNs, ANN-to-SNN conversion, and other distillation objectives.

For the final saved refit, the student objective is:

```text
loss = (1 - alpha) * MSE(student_prediction, target)
     + alpha * MSE(student_prediction, teacher_prediction)
```

The final alpha values for FD001-FD004 are `0.0`, `0.2`, `0.1`, and `0.2`. The final student prediction averages seeds 42, 43, and 44. Earlier experiments used additional hidden-state, attention, and spike-activity losses; those are not part of the final refit objective.

## Repository contents

- `preprocess_cmapss.py` - engine splits, train-fitted preprocessing, capped targets, and sequence windows.
- `train_baselines.py` - mean, Ridge, Random Forest, and DeepNN baselines.
- `train_sequence_baselines.py` - RNN, LSTM, GRU, BiLSTM, and CNN-LSTM baselines.
- `train_bilstm_teacher.py` - attention-BiLSTM teacher.
- `train_spike_models.py` - direct SNN, ANN-to-SNN, KD-LSTM, and initial SpikeDistill experiments.
- `tune_*.py`, `refine_*.py`, and `improve_spikedistill.py` - validation-driven experiments and refinements.
- `run_ablation_study.py` - loss/architecture ablations.
- `final_refit_eval.py` - selected full-training refit and official-test evaluation.
- `evaluate_*.py` and `select_snn_ensemble_weights.py` - additional evaluation workflows.
- `explain_spikedistill.py` - validation-only model interpretation outputs.
- `make_report_figures.py` - graphs from saved evaluation artifacts.
- `SpikeDistill_RUL_Reproducible_Workflow.ipynb` - notebook documenting the end-to-end workflow.

## Setup

Use Python 3.10 or newer. Create and activate an environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch will use CUDA when an available CUDA-enabled installation is present; otherwise the scripts run on CPU. For GPU use, install the PyTorch build appropriate for your CUDA setup before running the project.

## Dataset setup

Download the NASA C-MAPSS dataset through the official [NASA Open Data Portal](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data) and follow its terms. Place the extracted text files in `CMAPSSData/` so the folder contains:

```text
CMAPSSData/
  train_FD001.txt ... train_FD004.txt
  test_FD001.txt  ... test_FD004.txt
  RUL_FD001.txt   ... RUL_FD004.txt
```

The preprocessing script creates `CMAPSSData/processed/`. Raw data and generated files are excluded by `.gitignore`.

## Workflow

Run commands from the repository root.

1. Prepare windows and train/validation splits:

   ```bash
   python preprocess_cmapss.py
   ```

2. Train baseline models:

   ```bash
   python train_baselines.py
   python train_sequence_baselines.py
   ```

3. Train the teacher and initial student experiments:

   ```bash
   python train_bilstm_teacher.py
   python train_spike_models.py --stage all
   ```

4. Run validation refinements as needed. The scripts `tune_spikedistill.py`, `refine_spikedistill.py`, `improve_spikedistill.py`, and `refine_fd003.py` contain the project’s successive tuning experiments. Check each script’s arguments before launching. These experiments can be computationally intensive.

5. After the selected recipe/history artifacts have been generated, run the final refit and evaluation:

   ```bash
   python final_refit_eval.py
   ```

6. Optional validation explanations and figure generation:

   ```bash
   python explain_spikedistill.py
   python make_report_figures.py
   ```

Some evaluation and plotting scripts require their corresponding trained checkpoints and saved prediction artifacts. Run them only after the relevant workflow stages have produced those files.

The notebook presents this sequence in one place. It launches scripts as subprocesses and can run for a long time, especially during tuning.

## Evaluation summary

The saved final-refit results are mixed. SNN ensemble / teacher RMSE (cycles) is:

| Subset | Teacher | SNN ensemble |
| --- | ---: | ---: |
| FD001 | 14.141 | 16.328 |
| FD002 | 18.003 | 16.021 |
| FD003 | 17.181 | 21.692 |
| FD004 | 19.185 | 19.888 |

The SNN has lower RMSE only on FD002. Its C-MAPSS score is lower on FD002 and FD004, and higher on FD001 and FD003. Results and limitations should be reported subset by subset.

**Evaluation caveat:** the final-refit script records recipe selection using validation before test scoring. Other project artifacts note that official-test labels were inspected during earlier project iterations. Treat the saved metrics as the result of this project’s final-refit run, not as a completely untouched blind-test estimate.

The project includes a Jetson benchmark script, but no completed device measurements are included here. The code and spike rates do not by themselves establish lower latency, power, or energy use.

## Outputs and reproducibility

Scripts write metrics, predictions, checkpoints, and figures under `results/` and processed windows under `CMAPSSData/processed/`; those generated artifacts are intentionally ignored by Git. Save the run configuration and software/device details when reproducing results. Tuning must use train/validation data; do not use official-test scores to choose new models or hyperparameters.

## References

- NASA, [C-MAPSS Aircraft Engine Simulator Data](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data).
- A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,” 2008.
- E. O. Neftci, H. Mostafa, and F. Zenke, “Surrogate Gradient Learning in Spiking Neural Networks,” *IEEE Signal Processing Magazine*, 2019.
- G. Hinton, O. Vinyals, and J. Dean, “Distilling the Knowledge in a Neural Network,” 2015.
