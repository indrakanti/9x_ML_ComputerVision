# Module 26 — Contrastive Learning and CLIP-Style Image–Text Alignment

Contrastive learning maps related inputs close together in an embedding space and unrelated inputs farther apart.

CLIP-style training aligns image and text representations:

```text
image ─→ image encoder ─→ normalized image embedding
                                      │
                                      │ cosine similarity
                                      │
text  ─→ text encoder  ─→ normalized text embedding
```

## Learning objectives

You should be able to:

- explain paired image/text contrastive learning
- L2-normalize embeddings
- compute cosine similarity
- apply a temperature/logit scale
- construct an image-to-text similarity matrix
- explain symmetric image→text and text→image losses
- perform zero-shot/retrieval ranking
- reproduce similarity/ranking in C++17
- understand embedding-cache and precision tradeoffs

## 1. L2 normalization

For embedding $x$:

$$
\hat{x}=\frac{x}{\|x\|_2}
$$

After normalization:

$$
\hat{x}^T\hat{y}
$$

is cosine similarity.

## 2. Similarity matrix

For a batch of image embeddings $I$ and text embeddings $T$:

$$
S=\hat{I}\hat{T}^T
$$

Each row scores one image against every text candidate.

## 3. Temperature

CLIP-style systems scale similarity logits:

$$
L=\frac{S}{\tau}
$$

or equivalently multiply by a learned logit scale.

Smaller $\tau$ sharpens the softmax distribution.

## 4. Contrastive objective

For paired batches, the diagonal entries are positives.

Image-to-text cross entropy:

$$
\mathcal{L}_{i2t}
=
CE(L, [0,1,\dots,N-1])
$$

Text-to-image uses the transposed matrix.

The symmetric objective is:

$$
\mathcal{L}
=
\frac12(
\mathcal{L}_{i2t}
+
\mathcal{L}_{t2i}
)
$$

## 5. Zero-shot classification

If text prompts describe classes:

```text
image embedding
      ↓
similarity against class-text embeddings
      ↓
highest similarity
      ↓
predicted class
```

Prompt wording therefore becomes part of the model/evaluation contract.

## 6. Retrieval

The same embedding space supports:

- image → text retrieval
- text → image retrieval
- nearest-neighbor search
- multimodal indexing

At large scale, exact matrix multiplication may be replaced by approximate nearest-neighbor search.

## 7. Executable reference

The artifact contains:

- three image embeddings
- three text embeddings
- embedding dimension
- temperature

Python computes:

- normalized embeddings
- cosine similarity matrix
- temperature-scaled logits
- top text match for each image
- symmetric contrastive loss

PyTorch independently verifies the same loss.

C++17 reproduces the similarity matrix and retrieval ranking.

## 8. Deployment implications

Embedding systems introduce different runtime patterns from dense perception:

- encoder inference
- embedding normalization
- embedding cache/storage
- matrix multiplication or ANN search
- prompt/template versioning
- precision vs retrieval-quality tradeoffs

Text embeddings for a fixed class vocabulary can often be cached.

## 9. Failure modes

- forgetting normalization
- changing embedding dimension/order
- using distance instead of similarity inconsistently
- applying the wrong temperature direction
- mixing prompt sets across evaluations
- assuming embedding cosine scores are calibrated probabilities
- comparing retrieval metrics with different candidate pools

## 10. Engineering rules

1. version embedding model and dimension
2. normalize consistently
3. version prompt templates
4. preserve temperature/logit-scale semantics
5. compare complete similarity matrices
6. cache static text embeddings when valid
7. separate ranking quality from probability calibration

## 11. AV relevance

Vision-language embeddings can support open-vocabulary labeling, semantic search over driving scenes, dataset mining, long-tail discovery, and later open-vocabulary detection/grounding modules.

## 12. Book takeaway

> **Contrastive learning replaces a fixed classifier with a shared embedding geometry. Once image and text live in the same space, classification becomes similarity search—but the normalization, prompt, and temperature contracts become part of the system.**

## Reproducible commands

```bash
python 26_ContrastiveCLIP/python/clip_walkthrough.py
pytest 26_ContrastiveCLIP/tests
cmake -S 26_ContrastiveCLIP -B build-m26 -G Ninja \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-m26
ctest --test-dir build-m26 --output-on-failure
python 26_ContrastiveCLIP/python/clip_native_parity.py \
  build-m26/cv9x_clip_similarity \
  26_ContrastiveCLIP/artifacts/embeddings.txt
```

## Next

Module 27 introduces masked-image modeling.
