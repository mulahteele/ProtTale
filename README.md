# ProtTale

**ProtTale** is a protein function generation framework trained with **pairwise supervision**.  
Given a protein sequence, ProtTale generates free-text functional descriptions by aligning protein representations with textual function embeddings.

---

## Method Overview

ProtTale is trained with **pairwise protein–function supervision**, encouraging function embeddings to preserve semantic similarity between proteins.

Given two proteins \( P_i, P_j \) with similarity score \( s_{ij} \), the model learns function representations \( f_i, f_j \) such that:

\[
\mathcal{L}_{pair} = \left\| \langle f_i, f_j \rangle - s_{ij} \right\|_2^2
\]

At inference time, ProtTale generates function descriptions autoregressively:

\[
p(y \mid x) = \prod_{t=1}^{T} p(y_t \mid y_{<t}, x)
\]

where \( x \) is the protein sequence and \( y \) is the generated function text.

---

## Environment Setup

### Installation

1. **Create the Conda environment**
```bash
conda env create -n ProtTale -f ProtT3_environment.yml
