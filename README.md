# Continuous Difficulty Control for Text Simplification

This repository contains the core implementation of a two-stage method for continuous difficulty control in English text simplification.

Intermediate target values such as `2.5` and `3.5` can also be used for controlled generation.

## Required Files

| File | Purpose |
|---|---|
| `train_difficulty_estimator.py` | Stage 1: trains difficulty representations from CEFR-labeled sentences |
| `run_simplification_training.py` | Stage 2: trains the difficulty-controlled text simplification model |
| `estimate_difficulty.py` | Estimates the continuous difficulty of input sentences |
| `controllable_gen.py` | Generates sentences at a specified target difficulty |
| `models.py` | Defines the BART-based models |
| `preprocess.py` | Loads and preprocesses the data |
| `utils.py` | Provides shared configuration and utility functions |
