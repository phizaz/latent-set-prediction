# Official implementation of Set Prediction in the Latent Space (LSP)

A NeurIPS2021 paper:

```
@inproceedings{NEURIPS2021_d61e9e58,
 author = {Preechakul, Konpat and Piansaddhayanon, Chawan and Naowarat, Burin and Khandhawit, Tirasan and Sriswasdi, Sira and Chuangsuwanich, Ekapol},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {25516--25527},
 publisher = {Curran Associates, Inc.},
 title = {Set Prediction in the Latent Space},
 url = {https://proceedings.neurips.cc/paper/2021/file/d61e9e58ae1058322bc169943b39f1d8-Paper.pdf},
 volume = {34},
 year = {2021}
}
```

# Usage

Ready to use LSP (latent set prediction) code is available in `lsp.py`.
It's self-contained and annotated with comments using the convention from the published paper.

```
from lsp import LSPLoss

...

# assume a set prediction model & encoder model
S = set_prediction(x)
G = encoder(gt)

# where
# S = set elements (n, c)
# len_S = cardinalities of sets in a batch
# G, len_G should be the same size as S

# LSP latent loss
# default params
loss_fn = LSPLoss('gcr', w_loss_gs=1, w_loss_sg=0.1, d=1e-3)
latent = loss_fn(S, len_S, G, len_G)

# return values contain
# - S_pi = ordered set elements, used for loss calculation to allow proper gradient flow
# - S_i = index of the ordering such that S[S_i] == S_pi
# - loss = latent loss

# feeding the ordered set elements to the prediction head
# while allowing the gradient to flow through LSP correctly
pred = prediction_head(latent.S_pi)
total_loss = task_loss(pred, gt) + latent.loss

total_loss.backward()
```

# Reprodibility

We included reproducible code for the main experiments including:

- [CLEVR object description prediction task](image_captioning)
- [MIMIC-CXR chest radiograph report generation task](image_captioning)
- [MNIST object detection task](object_detection)