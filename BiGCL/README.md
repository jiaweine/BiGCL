# BiGCL: Semantic-Aware Image Clustering via VLM-based Bipartite Graph Contrastive Learning

## Overview

**BiGCL** proposes a novel framework that bridges Vision-Language Models (VLMs) and graph-based contrastive learning for unsupervised image clustering. Unlike prior CLIP-based clustering methods that treat VLM features as flat representations, BiGCL explicitly models the structural relationships between images and semantic concepts through a **bipartite graph**, where images and learnable semantic prototypes form two disjoint node sets connected by adaptive affinity edges.

### Key Contributions

1. **VLM-based Semantic Prototype Generation**: Learnable continuous prompts are decoded through a frozen CLIP text encoder to produce K semantic prototype embeddings, providing language-grounded cluster representations without requiring class names.

2. **Bipartite Graph Message Passing Network (BGMPN)**: A heterogeneous GNN that performs bidirectional cross-attention message passing between image nodes and prototype nodes, enabling mutual refinement of visual and semantic representations.

3. **Multi-Granularity Contrastive Learning**: Four complementary objectives operating at different levels:
   - **Node-level InfoNCE**: Instance discrimination between augmented views
   - **Cross-type Alignment**: Image-prototype contrastive matching via soft assignments
   - **Graph-level Contrastive**: Global readout matching between augmented graph views
   - **Prototype Uniformity**: Hyperspherical spreading of prototypes (Wang & Isola, ICML 2020)

4. **Progressive Self-Training**: Confidence-based pseudo-labeling with curriculum threshold scheduling for iterative cluster refinement.

## Architecture

```
Input Images
    |
    v
CLIP Visual Encoder (frozen) ---> Raw Visual Features
    |
    v
Visual Projector (trainable) ---> Projected Image Features (N x D)
    |                                        |
    |     Prompt Learner (learnable)         |
    |         |                              |
    |         v                              |
    |     CLIP Text Encoder (frozen)         |
    |         |                              |
    |         v                              |
    |     Semantic Prototypes (K x D) -------+
    |                                        |
    v                                        v
    +-----> Bipartite Graph Construction <---+
                    |
                    v
            BGMPN (L layers of bidirectional cross-attention)
                    |
            +-------+-------+
            |               |
            v               v
    Refined Image      Refined Proto
    Features           Features
            |               |
            v               v
    Graph Augmentor x2 (feature drop + edge drop)
            |               |
            v               v
    Multi-Granularity Contrastive Loss
            |
            v
    Clustering Head (Sinkhorn-normalized assignments)
            |
            v
    Progressive Self-Training (confidence-based pseudo-labels)
```

## Method Details

### Bipartite Graph Construction

Given N image features and K prototype features, we construct a bipartite adjacency matrix A in R^{N x K} via:

1. Compute cosine similarity: S = norm(V) @ norm(P)^T
2. Top-k sparsification per image node
3. Temperature-scaled softmax over surviving entries

The adjacency is **rebuilt at each GNN layer** using updated features, allowing the graph structure to co-evolve with representations.

### Bipartite Graph Message Passing

Each BGMPN layer performs two phases:
- **Image-to-Prototype**: Prototypes query image neighbors via multi-head cross-attention
- **Prototype-to-Image**: Images query prototype neighbors via multi-head cross-attention

Both phases use residual connections, layer normalization, and feed-forward networks.

### Loss Functions

The total training objective is:

```
L = w1 * L_node + w2 * L_cross + w3 * L_graph + w4 * L_uniform + w5 * L_entropy + w6 * L_self
```

| Loss | Description | Default Weight |
|------|-------------|----------------|
| L_node | InfoNCE between augmented image views | 1.0 |
| L_cross | Cross-type alignment with soft assignments | 1.0 |
| L_graph | Graph-level readout contrastive | 0.5 |
| L_uniform | Prototype uniformity on hypersphere | 0.5 |
| L_entropy | Marginal entropy max + conditional entropy min | 0.5 |
| L_self | Cross-entropy on confident pseudo-labels | 1.0 |

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: PyTorch >= 2.0, open-clip-torch, einops, scikit-learn, scipy, PyYAML, tqdm

## Usage

### Training

```bash
# CIFAR-10
python train.py --config configs/cifar10.yaml --gpu 0

# CIFAR-100 (20 superclasses)
python train.py --config configs/cifar100.yaml --gpu 0

# STL-10
python train.py --config configs/stl10.yaml --gpu 0

# ImageNet-R (ViT-L/14 backbone)
python train.py --config configs/imagenet_r.yaml --gpu 0

# Custom overrides
python train.py --config configs/cifar10.yaml --lr 5e-4 --batch_size 128 --epochs 300
```

### Evaluation

```bash
python eval.py --checkpoint output/cifar10_k10_s42/best.pth --save_features
```

### Multi-seed Evaluation

```bash
for seed in 42 123 456; do
    python train.py --config configs/cifar10.yaml --seed $seed
done
```

## Project Structure

```
BiGCL/
├── configs/                  # YAML configuration files
│   ├── default.yaml          # Default hyperparameters
│   ├── cifar10.yaml
│   ├── cifar100.yaml
│   ├── stl10.yaml
│   └── imagenet_r.yaml
├── models/
│   ├── bigcl.py              # Main BiGCL model
│   ├── clip_backbone.py      # CLIP feature extraction
│   ├── prompt_learner.py     # Learnable prompt module
│   ├── bipartite_gnn.py      # Bipartite graph neural network
│   ├── clustering_head.py    # Clustering + self-training
│   └── losses.py             # Multi-granularity contrastive losses
├── data/
│   ├── datasets.py           # Dataset loaders
│   └── augmentations.py      # CLIP-compatible augmentations
├── utils/
│   ├── metrics.py            # ACC, NMI, ARI
│   ├── graph_utils.py        # Graph construction utilities
│   └── misc.py               # Seeds, checkpoints, schedulers
├── train.py                  # Training entry point
├── eval.py                   # Evaluation entry point
├── requirements.txt
└── README.md
```

## Benchmarks

| Dataset | #Samples | #Classes | CLIP Backbone |
|---------|----------|----------|---------------|
| CIFAR-10 | 60,000 | 10 | ViT-B/32 |
| CIFAR-100-20 | 60,000 | 20 | ViT-B/32 |
| STL-10 | 13,000 | 10 | ViT-B/32 |
| ImageNet-R | 30,000 | 200 | ViT-L/14 |

## Related Work

- **CLIP** (Radford et al., ICML 2021): Contrastive Language-Image Pre-training
- **TEMI** (Adaloglou et al., BMVC 2023): Teacher-student framework for CLIP-based clustering
- **SPICE** (Niu et al., CVPR 2022): Semantic pseudo-labeling for image clustering
- **PromptCAL** (Zhang et al., CVPR 2023): Prompt-based contrastive alignment learning
- **CoOp** (Zhou et al., IJCV 2022): Context optimization for prompt learning
- **SwAV** (Caron et al., NeurIPS 2020): Sinkhorn-based online clustering
- **Wang & Isola** (ICML 2020): Understanding contrastive representation learning through alignment and uniformity

## Citation

```bibtex
@inproceedings{bigcl2025,
    title={Semantic-Aware Image Clustering via VLM-based Bipartite Graph Contrastive Learning},
    author={Anonymous},
    booktitle={Proceedings of Conference},
    year={2025}
}
```

## License

This project is released under the MIT License.
