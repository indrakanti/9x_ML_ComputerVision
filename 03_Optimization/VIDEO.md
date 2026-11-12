# Episode 03 — Optimization: Losses, SGD, Momentum & Adam

Suggested title:

**How Neural Networks Actually Learn — SGD, Momentum and Adam**

Suggested thumbnail:

**GRADIENT ≠ OPTIMIZER**

Target duration: 35–45 minutes.

## Chapters

~~~text
00:00 What an optimizer actually does
04:00 Objective functions and gradients
09:00 Gradient descent from first principles
15:00 Learning-rate failure modes
20:00 SGD and minibatch noise
25:00 Momentum
31:00 Adam
37:00 Manual optimizer vs PyTorch
41:00 What to log in real training
~~~

## Demonstration

~~~bash
python 03_Optimization/python/optimization_walkthrough.py
pytest 03_Optimization/tests -q
~~~

## Core message

Training is a repeated numerical optimization process:

~~~text
loss
 ↓
gradient
 ↓
optimizer state
 ↓
parameter update
~~~

A model can have correct math and still fail because optimization dynamics are poor.
