(For a more in-depth wiki, check https://deepwiki.com/gems-uff/sbcr)

# SBCR - Source Code Conflict Resolution

A research system that implements Iterated Local Search (ILS) algorithms for automated resolution of merge conflicts in source code. `sbcr.py:1`

## Overview

SBCR (Source Code Conflict Resolution) addresses the problem of automatically resolving source code merge conflicts by combining lines from two conflicting versions (v1 and v2) while preserving partial order constraints. The system uses a Gestalt similarity metric based on Longest Common Subsequence (LCS) to evaluate candidate resolutions and employs ILS metaheuristics to search for optimal conflict resolutions. `sbcr.py:303-327`

This repository is structured as a three-tier research system designed for academic research and includes comprehensive experimental infrastructure for evaluating conflict resolution approaches.

## Features

- **Automated Conflict Resolution**: Uses Iterated Local Search to generate optimal merge conflict resolutions
- **Gestalt Similarity Evaluation**: Employs LCS-based similarity metrics for candidate assessment `sbcr.py:62-83`
- **Experimental Infrastructure**: Comprehensive batch evaluation and parameter tuning systems
- **Research Analysis**: Jupyter notebooks for statistical analysis and performance evaluation
- **Configurable Parameters**: Multiple timeout and optimization parameters for algorithm tuning `sbcr.py:7-13`

## Repository Structure

├── sbcr.py # Core ILS algorithm implementation
├── TOSEM2025/
│ ├── sbcr_evaluate.py # Batch evaluation system
│ ├── sbcr_tunning.py # Parameter tuning infrastructure
│ └── analyses_notebooks/ # Research analysis notebooks
│ ├── RQ1.ipynb # Dataset characteristics analysis
│ ├── RQ2.ipynb # Random candidate similarity analysis
│ ├── RQ5.ipynb # Aggregation function evaluation
│ ├── RQ6.ipynb # Performance analysis
│ └── RQ6_tuning.ipynb # Parameter tuning results


## Installation

### Dependencies

The system requires the following Python packages:

- `pandas` - Data manipulation and Excel output
- `pylcs` - Efficient LCS computation for Gestalt similarity `sbcr.py:3`
- `matplotlib` - Visualization for research analysis
- `scipy` - Statistical analysis functions
- `jupyter` - For running analysis notebooks

Install dependencies:
```
pip install pandas pylcs matplotlib scipy jupyter
```

## Usage

### Basic Command Line Usage

Run the core algorithm on two conflicting files:
```
python sbcr.py path/to/v1_file.txt path/to/v2_file.txt
```

### Batch Evaluation

For systematic evaluation across multiple conflict instances:
```
python TOSEM2025/sbcr_evaluate.py
```


### Parameter Tuning

To optimize algorithm parameters: 
```
python TOSEM2025/sbcr_tunning.py
```


## Algorithm Overview

The SBCR algorithm implements an Iterated Local Search approach with the following key components:

### Candidate Representation

Candidates are represented as tuples containing source version and line index:
```
[('v1', 0), ('v1', 1), ('v2', 1), ('v1', 2)]
```


### Fitness Evaluation

Uses Gestalt similarity based on Longest Common Subsequence:

- **Formula**: 2 * len(LCS) / (len(text1) + len(text2))
- **Fitness**: Average similarity with both input versions

### Local Search Operations

Three neighborhood operations maintain partial order constraints:

- **Remove**: Delete a line from candidate
- **Swap**: Exchange adjacent lines from different versions
- **Add**: Insert a new line while preserving order 

### Configuration Parameters

Key algorithm parameters and their defaults: 

| Parameter                                          | Default | Description                                      |
|----------------------------------------------------|---------|--------------------------------------------------|
| ILS_TIMEOUT_SECONDS                                | 15      | Maximum ILS execution time                       |
| LOCAL_SEARCH_MAXIMUM_NEIGHBORS                     | 5       | Maximum neighbors per local search               |
| ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT   | 10      | Early stopping criterion                         |
| NEIGHBOR_FINDING_TIMEOUT_SECONDS                   | 3       | Timeout for neighbor generation                  |
| RANDOM_SEED                                        | 3022024 | Reproducibility seed                             |

## Research Components

### Research Questions

The analysis framework evaluates SBCR through six research questions:

- **RQ1**: Dataset characteristics and conflict composition analysis
- **RQ2**: Random candidate similarity to parent versions and resolutions
- **RQ3**: Aggregation functions impact on correlations between resolutions and parent versions
- **RQ4**: Analysis of corner cases
- **RQ5**: Comprehensive aggregation function evaluation
- **RQ6**: Performance analysis and parameter tuning results

### Data Processing Pipeline

The system processes JSON datasets containing merge conflict chunks and generates comprehensive results for academic analysis.

### Statistical Analysis

Each notebook implements systematic correlation analysis using `scipy.stats` functions to evaluate relationships between similarity measures and developer resolutions.

## Academic Use

This system is designed for academic research on automated conflict resolution. It includes:

- Comprehensive experimental infrastructure for reproducible research
- Statistical analysis notebooks for research publication
- Parameter tuning systems for optimization studies
- Performance tracking and metrics collection

## Output Formats

The system generates multiple output artifacts:

- Candidate resolutions in text format
- Performance metrics in Excel spreadsheets
- Iteration tracking data for algorithm analysis
- Statistical summaries for research publication

## Notes

- The algorithm ensures generated candidates are not identical to either input version, forcing actual conflict resolution rather than simple selection 
- Perturbation generates entirely new random candidates to escape local optima rather than making small modifications 
- The system implements multi-level timeouts (neighbor generation, local search, overall ILS) to ensure robust termination
- All analysis notebooks follow consistent data loading patterns and use standardized column schemas for reproducibility

## License

This project is designed for academic research purposes. Please cite appropriately if used in research publications.







