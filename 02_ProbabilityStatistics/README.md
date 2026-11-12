# Module 02 — Probability & Statistics for Vision ML

This module teaches the probability and statistics needed to understand training, evaluation, uncertainty and data behavior in later computer-vision models.

## Learning objectives

By the end of this module you should be able to:

- compute and interpret mean, variance and standard deviation
- distinguish variance from covariance and correlation
- explain discrete vs continuous probability distributions
- reason about conditional probability and Bayes' rule
- understand likelihood and why it appears in model training
- connect Gaussian assumptions to common losses
- estimate confidence intervals for measured accuracy
- understand sampling noise and why one metric value is not absolute truth
- recognize class imbalance, calibration and uncertainty as statistical problems

## 1. Why statistics matters in vision

A model is trained from finite data and evaluated from finite samples.

That means every measured quantity is influenced by uncertainty:

~~~text
dataset sampling
label noise
sensor noise
class imbalance
model stochasticity
augmentation
optimization
measurement noise
~~~

Probability gives us a language for uncertain events.

Statistics gives us tools for estimating properties from observed data.

## 2. Mean and variance

For samples $x_1, x_2, \ldots, x_n$, the sample mean is:

$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
$

Population variance is:

$
\sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2
$

Sample variance commonly uses Bessel's correction:

$
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2
$

Standard deviation is:

$
\sigma=\sqrt{\sigma^2}
$

### Vision example

Pixel intensities with the same mean can still have very different contrast because their variances differ.

## 3. Covariance and correlation

Covariance measures how two variables vary together:

$
\operatorname{Cov}(X,Y)=
\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]
$

Correlation normalizes covariance:

$
\rho_{XY}=
\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}
$

Correlation is bounded between -1 and 1.

### Vision example

RGB channels are often statistically correlated. Feature dimensions can also become strongly correlated, which is one reason normalization and whitening concepts appear in ML.

## 4. Probability distributions

Important distributions for this course include:

- Bernoulli — binary events
- Binomial — counts of successes
- Categorical — one of several classes
- Gaussian — continuous noisy measurements
- Uniform — equal probability over a range

For a Gaussian:

$
p(x)=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
$

The parameters are mean $\mu$ and variance $\sigma^2$.

## 5. Conditional probability

Conditional probability asks:

> What is the probability of A given that B is already known?

$
P(A|B)=\frac{P(A\cap B)}{P(B)}
$

Bayes' rule is:

$
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
$

### Vision example

Suppose a detector predicts "pedestrian."

Bayes tells us that the posterior probability depends on:

~~~text
how often the detector fires for pedestrians
how often it fires falsely
how common pedestrians are in the environment
~~~

That last item is the prior.

## 6. Likelihood

Probability and likelihood use the same mathematical model but ask different questions.

Probability:
> Given parameters, how probable is the data?

Likelihood:
> Given observed data, how plausible are the parameters?

For observations (x_1,ldots,x_n) and parameters (	heta):

[
L(	heta)
=
prod_{i=1}^{n}p(x_i|	heta)
]

We usually maximize log-likelihood:

[
log L(	heta)
=
sum_{i=1}^{n}log p(x_i|	heta)
]

because sums are numerically easier than long products.

## 7. Gaussian likelihood and squared error

If residual errors are assumed Gaussian:

$
e_i \sim \mathcal{N}(0,\sigma^2)
$

maximizing likelihood leads to minimizing squared error:

$
\sum_i e_i^2
$

This statistical assumption explains why MSE appears so often.

Loss functions are not arbitrary formulas. They often correspond to probabilistic assumptions.

## 8. Bernoulli likelihood and binary cross-entropy

For a binary target $y\in\{0,1\}$ and predicted probability $p$:

$
P(y|p)=p^y(1-p)^{1-y}
$

Negative log-likelihood gives:

$
-\log P(y|p)
=
-y\log p -(1-y)\log(1-p)
$

This is binary cross-entropy.

Later modules will derive classification losses in more depth.

## 9. Sampling uncertainty

Suppose a classifier is correct on 90 out of 100 test images.

Measured accuracy:

$
\hat{p}=0.90
$

But this is an estimate of an underlying probability, not absolute truth.

A normal-approximation standard error is:

$
SE=\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$

An approximate 95% interval is:

$
\hat{p}\pm1.96SE
$

This becomes important when comparing two models whose metrics differ only slightly.

## 10. Runtime applicability

| Gate | Status | Why |
|---|---|---|
| Python formatting / lint | Required | Python implementation |
| mypy | Required | Typed statistics helpers |
| pytest | Required | Formula and behavior validation |
| CPU PyTorch tests | N/A | This module uses NumPy/statistics, not PyTorch behavior |
| C++ configure/build/CTest | Required | Native statistics and metric computation is production-relevant |
| Python ↔ C++ numerical parity | Required | Same deterministic vectors must match |
| ONNX tests | N/A | No model artifact |
| Python ↔ exported-model equivalence | N/A | No exported model |

## 11. Run the demonstrations

~~~bash
source .venv/bin/activate

python 02_ProbabilityStatistics/python/statistics_walkthrough.py
pytest 02_ProbabilityStatistics/tests -q

cmake -S 02_ProbabilityStatistics -B build/module02 -DCMAKE_BUILD_TYPE=Release
cmake --build build/module02 --parallel
ctest --test-dir build/module02 --output-on-failure

python 02_ProbabilityStatistics/python/native_parity.py \
  build/module02/cv9x_statistics
~~~

## 12. Native C++ counterpart

The C++17 implementation provides deterministic native versions of:

- mean
- population variance
- covariance
- correlation
- accuracy confidence interval

The native executable exposes both `--self-test` and `--json`. The Python parity checker executes the compiled binary and compares every reported value against the Python reference at a strict numerical tolerance.

This establishes an early course pattern:

```text
Python reference
      ↓
C++17 implementation
      ↓
numerical parity evidence
```

## 13. What the walkthrough demonstrates

- image-intensity mean and variance
- covariance and correlation
- Gaussian sampling
- empirical probability
- Bayes rule
- Bernoulli negative log-likelihood
- Gaussian negative log-likelihood relationship to squared error
- accuracy confidence intervals
- reproducible random sampling

## 14. Engineering rules

From this module onward:

1. always report sample size with metrics
2. do not treat tiny metric differences as meaningful without uncertainty context
3. distinguish probability from confidence score
4. record class frequencies
5. distinguish population statistics from sample estimates
6. seed stochastic experiments when reproducibility matters
7. do not infer causation from correlation

## 15. Failure modes

### Tiny validation set

A model can appear dramatically better or worse due to sampling noise.

### Class imbalance hidden by accuracy

A model can achieve high accuracy while failing a rare but important class.

### Correlation interpreted as causation

Two image variables may move together without one causing the other.

### Treating neural-network scores as calibrated probabilities

A softmax value is not automatically a well-calibrated probability.

### Leakage

If validation data influences training or preprocessing decisions incorrectly, measured metrics become statistically invalid.

## 16. Exercises

1. Generate two distributions with identical mean but different variance.
2. Create two correlated variables and measure covariance/correlation.
3. Simulate a classifier with 90% true accuracy at different dataset sizes.
4. Compare the width of confidence intervals at n=50, 500 and 5000.
5. Implement Bayes' rule for a rare-event detector.
6. Show numerically that Gaussian negative log-likelihood differs from squared error only by scale/constants when variance is fixed.
7. Create a highly imbalanced binary dataset and show why accuracy is misleading.

## Next

Module 03 introduces optimization: objective functions, gradients, SGD, momentum and Adam.
