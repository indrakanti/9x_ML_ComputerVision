# Module 34 — Visual Question Answering and Multimodal Reasoning

## Learning outcomes

- Explain how image patch features and a question embedding interact.
- Implement scaled dot-product grounding, attention normalization, and answer selection in **Python and C++17**.
- Distinguish a deterministic grounding illustration from a **trained VQA model**.
- Run optional real-image VQA using a pretrained ViLT model.

```text
Image → patch embeddings ──┐
                           ├─ question-conditioned attention → answer
Question → text embedding ─┘
```

The CPU fixture uses two patches, a two-dimensional question, and associated patch labels. It selects the label of the patch with the highest scaled dot-product attention. This is **not** a language-generating or reasoning VQA model. It illustrates the grounding mechanism, while the optional pretrained model provides real VQA behavior.

## Offline Python/C++ parity

```bash
python 02_Multimodal_3D_and_Generative_Vision/34_VisualQuestionAnswering/python/vqa.py --synthetic
pytest 02_Multimodal_3D_and_Generative_Vision/34_VisualQuestionAnswering/python
cmake -S . -B build-native -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-native --parallel
ctest --test-dir build-native --output-on-failure
```

## Optional applied example (pretrained model)

```bash
python -m pip install transformers pillow torch
python 02_Multimodal_3D_and_Generative_Vision/34_VisualQuestionAnswering/python/vqa.py \
  --image path/to/open-license-photo.jpg --question "What vehicle is visible?"
```

Default model: `dandelin/vilt-b32-finetuned-vqa`. The example downloads pretrained weights; do not commit weights or dataset images. Use appropriately licensed images. VQA answers may hallucinate; validate them against the image.

**Limitations:** the deterministic fixture is not a learned encoder, the optional ViLT model is not fine-tuned in this lab, and neither path demonstrates multi-hop reasoning or calibrated uncertainty. These are extensions for later modules.
