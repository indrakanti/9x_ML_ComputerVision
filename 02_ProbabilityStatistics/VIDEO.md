# Episode 02 — Probability & Statistics You Actually Need for Vision ML

Suggested title:

**Probability & Statistics for Computer Vision — Only What You Need**

Suggested thumbnail:

**WHY 90% IS NOT JUST 90%**

Target duration: 32–40 minutes.

## Chapters

~~~text
00:00 Why ML vision is statistical
03:00 Mean / variance / standard deviation
08:00 Covariance and correlation
13:00 Probability distributions
18:00 Conditional probability
22:00 Bayes rule with a detector example
27:00 Likelihood vs probability
31:00 Gaussian likelihood → squared error
35:00 Bernoulli likelihood → BCE
38:00 Accuracy uncertainty and confidence intervals
~~~

## Demonstration

~~~bash
python 02_ProbabilityStatistics/python/statistics_walkthrough.py
pytest 02_ProbabilityStatistics/tests -q
pytest -q
~~~

## Core message

A measured model metric is an estimate from finite data, not an absolute fact.

Probability and statistics explain:

~~~text
uncertainty
noise
loss functions
class imbalance
sampling
metric variation
confidence
~~~
