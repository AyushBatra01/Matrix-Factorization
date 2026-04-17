# Matrix-Factorization for Recipe Recommendation

## Overview

This project explores the use of matrix factorization for building a recipe recommendation system from implicit user feedback. Rather than relying on existing libraries, I implemented a custom training pipeline from scratch to better understand the optimization challenges involved in sparse recommendation settings.

While matrix factorization is a standard technique, training it effectively on implicit data proved significantly more difficult than expected. This project evolved into an investigation of why training fails. 

---

## Objectives

* Build a recommendation system from a sparse user–recipe interaction matrix
* Implement matrix factorization with implicit feedback
* Explore training strategies such as negative sampling and confidence weighting
* Analyze optimization behavior and failure modes

---

## Data

The dataset of recipes and user-recipe interactions comes from [Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews/data). 

* `recipes_sample.csv`: a subset of almost 10,000 recipes from Food.com
* `reviews_sample.csv`: a subset of over 136,000 recipe reviews from Food.com

---

## Model Implementation

The core model factorizes a sparse user–item matrix into low-dimensional embeddings:

* User embeddings: ( U \in \mathbb{R}^{n \times K} )
* Item embeddings: ( V \in \mathbb{R}^{m \times K} )

Predictions are made via:

[
\hat{y}_{ui} = \sigma(U_u^\top V_i)
]

where \sigma(x) is the sigmoid function.

### Key features

* Custom training loop
* Negative sampling for implicit feedback
* Confidence-weighted loss:

  * Higher weight for observed interactions
  * Tunable negative sample weighting
* Separate tracking of:

  * Total loss
  * Positive sample loss
  * Negative sample loss
* Gradient clipping and learning rate scheduling
* Manual train/validation masking per user
* Support for new user embedding inference

---

## Experiments

Several strategies were explored to improve training:

* Varying negative sampling ratios
* Popularity-based vs uniform negative sampling
* Separate learning rates for positive and negative samples
* Confidence weighting schemes
* Embedding normalization
* Gradient clipping thresholds

---

## Baseline

To contextualize performance, simple baselines were implemented:

* Popularity-based recommendation
* Random recommendation

These provide a reference point for evaluating whether the model learns meaningful structure.

---

## Results

Training behavior was often unstable:

* Loss curves were noisy or failed to decrease consistently
* Model performance frequently did not outperform simple baselines
* Positive and negative losses behaved very differently depending on weighting

---

## Analysis

The experiments suggest several challenges inherent to implicit-feedback matrix factorization:

* **Class imbalance**: extremely sparse positives vs abundant negatives
* **Sampling bias**: negative sampling strongly affects gradient estimates
* **Gradient scaling issues**: interaction between confidence weights and learning rates
* **Sigmoid saturation**: large dot products lead to vanishing gradients
* **Data sparsity**: limited signal per user makes optimization difficult

These factors make training highly sensitive to hyperparameters and implementation details.

---

## Lessons Learned

* Implementing recommender systems from scratch exposes non-obvious optimization issues
* Negative sampling is a critical and fragile component
* Loss behavior alone is not a reliable indicator of recommendation quality
* Simple baselines are surprisingly competitive in sparse settings

---

## Future Work

Potential improvements include:

* Bayesian Personalized Ranking (BPR) loss
* Alternating Least Squares (ALS) for implicit data
* PyTorch-based implementation for better optimization control
* More principled negative sampling strategies
* Incorporation of bias terms and regularization schemes

---

## Repository Structure

```
src/            # Matrix factorization implementation
notebooks/      # Experiments and analysis
data/           # processed datasets
```

---

## Takeaway

This project highlights that building a recommender system is not just about implementing an algorithm. It requires careful consideration of optimization, data sparsity, and evaluation. Even simple models can exhibit complex and unintuitive behavior when trained on real-world implicit data.
