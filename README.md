# Unsupervised Concept Drift Detection based on Parallel Activations of Neural Network

### Abstract

> Practical applications of artificial intelligence increasingly have to deal with the streaming properties of real data, which, considering the time factor, are subject to phenomena such as periodicity and more or less chaotic degeneration -- resulting directly in the *concept drifts*. The modern concept drift detectors almost always assume immediate access to labels, which due to their cost, limited availability and possible delay has been shown to be unrealistic. This work proposes an unsupervised *Parallel Activations Drift Detector*, utilizing the outputs of an untrained neural network, presenting its key design elements, intuitions about processing properties, and a pool of computer experiments demonstrating its competitiveness in relation to *state-of-the-art* methods.

<video width="320" height="240" autoplay>
  <source src="output.mp4" type="video/mp4">
</video>

## How to replicate experiments?

The experiment code is available in scripts `e1.py` for hyperparameter calibration and `e2.py` for method comparison. The rerefence methods are implemented in `reference` (unsupervised) and `detectors` (supervised) directories. 

To replicate the experiments, it is possible to run the experimental code of `e1.py` and `e2.py`. This will generate files in `res` directory.

## Organisation of a repository

### Main elements

- `e1.py` – main loop of preliminary experiment.
- `e2.py` - main loop of comparative experiment.

- `methods.py` – impementation of proposed PADD method.
- `detectors` directory – implementation of `MetaClassifier`, enabling processing of comparative experiments and supervised drift detectors (`ADWIN`, `DDM` and `EDDM`) adapted to chunk-based evaluation protocol.
- `reference` directory – implementations of unsupervised drift detectors (`CDDD`, `MD3` and `CDDD`).
- `utils.py` – functional implementation of employed drift detection metrics and set of helper functions for implementation of experiments.

- `analyze_e2.py` – analysis of comparative experiment preparing processing artifacts for interpretation.
- `cd_plots_e2.py` – a *Critical Difference* test for comparative experiment.
- `plot_e1.py` – script preparing visualization for Figure 1 in supplementary material.
- `plot_e1_reduced.py` – script preparing visualization for Figure 2 in supplementary material .
- `plot_e2.py` – script preparing visualization for Figures 3-6 in supplementary material and Figure 2 in main article.


## Processing artifacts

- `res` directory – storage for results of conducted experiments
- `fig` directory – illustrations for the paper and supplementary materials
