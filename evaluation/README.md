# Evaluation

This directory contains the evaluation artifacts for the `intro-deep-learning-starter` project.

The goal is to evaluate the notebook not only by whether the models run, but also by whether the experiments provide reproducible, interpretable, and technically meaningful evidence of the deep learning concepts demonstrated in the project.

## Evaluation Scope

The evaluation covers:

* Reproducibility of the experiments
* Data preprocessing and train/validation separation
* Regression model performance
* Binary classification performance
* Overfitting and underfitting behavior
* Early stopping effectiveness
* Dropout vs. Batch Normalization
* Training configuration and optimization choices
* Quality and interpretability of reported results
* Limitations and conclusions

## Evaluation Criteria

| Criterion          | Description                                                                |
| ------------------ | -------------------------------------------------------------------------- |
| Reproducibility    | Experiments use explicit random seeds and documented dependencies          |
| Data handling      | Scaling and train/validation splits avoid data leakage                     |
| Regression         | Performance is evaluated using appropriate regression metrics              |
| Classification     | Accuracy and additional classification metrics are considered              |
| Regularization     | Overfitting mitigation techniques are experimentally compared              |
| Optimization       | Optimizer, learning rate, batch size, and training behavior are documented |
| Interpretability   | Results are presented with meaningful explanations                         |
| Experimental rigor | Conclusions are based on observed validation results                       |
| Documentation      | Evaluation methodology and limitations are clearly documented              |

## Experiments Evaluated

### 1. Single Neuron

Evaluates a single `Dense(1)` model as a baseline for regression.

Primary metric:

* Validation MAE

### 2. Deep Neural Network

Evaluates whether a multilayer neural network improves upon the single-neuron baseline.

Primary metrics:

* Validation MAE
* RMSE
* R²

### 3. Stochastic Gradient Descent

Evaluates explicit optimizer configuration and training behavior.

The evaluation considers:

* Learning rate
* Batch size
* Training loss
* Validation loss
* Convergence behavior

### 4. Overfitting and Early Stopping

Evaluates whether a larger network demonstrates overfitting and whether `EarlyStopping` improves generalization.

The evaluation considers:

* Training/validation loss gap
* Number of training epochs
* Best validation performance
* Restored model weights

### 5. Dropout vs. Batch Normalization

Compares regularization strategies rather than assuming that one technique is universally better.

The evaluation considers:

* Training loss
* Validation loss
* Train/validation gap
* Generalization behavior
* Dataset-dependent effects

### 6. Binary Classification

Evaluates a neural network classifier using the breast cancer dataset.

Primary metrics:

* Validation accuracy
* Precision
* Recall
* F1 score
* ROC-AUC

## Reproducibility

The project uses explicit random seeds and documents the main software versions used during development.

The experiments should be considered reproducible when:

1. Dependencies can be installed successfully.
2. The notebook executes from beginning to end without errors.
3. The datasets are obtained through `scikit-learn`.
4. The same preprocessing pipeline is applied consistently.
5. Results remain reasonably consistent across executions.

Exact numerical results may vary slightly depending on the TensorFlow/Keras version, hardware, and execution environment.

## Evaluation Report

The detailed evaluation is documented in:

* [`evaluation_report.md`](evaluation_report.md)

Machine-readable results are stored in:

* [`results.json`](results.json)

## Limitations

This evaluation is intended to assess a small educational deep learning project rather than a production ML system.

Important limitations include:

* Small datasets
* Single train/validation split
* No external test set
* Limited hyperparameter search
* No production deployment
* No experiment tracking system
* No automated model registry
* Results may vary across environments

Therefore, the reported metrics should be interpreted as educational experimental results rather than production-grade benchmark results.

## Success Criteria

The project is considered successful when it demonstrates that:

* Fundamental neural network concepts can be implemented with TensorFlow/Keras.
* Regression and binary classification workflows are correctly constructed.
* Training and validation behavior can be interpreted.
* Overfitting can be demonstrated and mitigated.
* Regularization techniques can be compared empirically.
* Conclusions are supported by observed results rather than assumptions.
* The experiments are reproducible from the repository.

## Overall Evaluation

The evaluation prioritizes **technical correctness, reproducibility, experimental reasoning, and educational value** over model complexity.

The project should demonstrate not simply that a neural network can be trained, but that the developer understands **why the models behave as they do and how training decisions affect generalization**.
