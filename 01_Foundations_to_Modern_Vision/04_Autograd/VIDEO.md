# Episode 04 — Autograd from First Principles

Suggested title:

**Build Autograd from Scratch — Then Compare It with PyTorch**

Suggested thumbnail:

**BACKPROP = CHAIN RULE**

Target duration: 35–45 minutes.

## Chapters

~~~text
00:00 Why automatic differentiation exists
04:00 Forward graph vs backward graph
09:00 Chain rule
15:00 Local derivatives
21:00 Reverse-mode autodiff
28:00 Gradient accumulation
33:00 Numerical gradient checking
38:00 PyTorch autograd mapping
42:00 Common failure modes
~~~

## Demonstration

~~~bash
python 04_Autograd/python/autograd_walkthrough.py
pytest 04_Autograd/tests -q
~~~

## Core message

Autograd is not magic. It is systematic application of local derivatives and the chain rule over a computation graph.
