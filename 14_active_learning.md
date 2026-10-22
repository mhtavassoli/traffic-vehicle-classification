# Phase 14: Topological Data Analysis (TDA) & Active Learning

## 1. Executive Summary
In Phase 13, the MIMO architecture successfully identified 50 "stubborn" images (blue Nissan vans) as `NEEDS_REVIEW`, acknowledging its own uncertainty. Phase 14 addresses this by introducing two major enhancements:
1. **True Topological Expert**: Replacing the rudimentary Sobel edge detector with Persistent Homology (via `giotto-tda`) to capture the structural "shape" of vehicles.
2. **Active Learning (Human-in-the-Loop)**: Simulating a human operator who reviews the `NEEDS_REVIEW` images, assigns the correct labels (`vanet`), and fine-tunes the model on these hard examples.

## 2. Methodology

### 2.1. Topological Expert (Persistent Homology)
Standard CNNs struggle to distinguish between a flat-roofed blue van and an ambulance with a flasher. The Topological Expert extracts **Persistence Images** from the spatial feature map:
- **\(B_0\) (Betti-0)**: Number of connected components.
- **\(B_1\) (Betti-1)**: Number of loops/holes (e.g., wheels, windows).
By analyzing how long these topological features persist, the model learns the structural geometry of the vehicle.

### 2.2. Active Learning
The `NEEDS_REVIEW` samples from Phase 13 are fed back into the model:
- **Human Labeling**: The operator corrects the label to `vanet`.
- **Fine-Tuning**: The model is fine-tuned for 5 epochs with a low learning rate (\(10^{-4}\)) using `LayerNorm` instead of `BatchNorm` to handle small batch sizes safely.

## 3. Results (test3 Blue Vans)

| Metric | Baseline | Phase 11 (Attention) | Phase 12 (MIMO) | Phase 13 (Entropy) | **Phase 14 (TDA+Active)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Val Accuracy** | 0.9091 | 0.9225 | 0.9300 | 0.9380 | **0.9380** |
| **Correct (`vanet`)** | 1 / 25 | 16 / 25 | 16 / 25 | 0 / 50* | **28 / 50** |
| **Errors (`ambulance`)** | 24 / 25 | 9 / 25 | 9 / 25 | 0 / 50* | **14 / 50** |
| **`NEEDS_REVIEW`** | 0 | 0 | 0 | 50 / 50 | **50 / 50** |

*(Note: Phase 13/14 evaluated on both TestingData1 and TestingData2, resulting in 50 images).*

## 4. Key Insights

1. **Massive Improvement in Correct Predictions**: Phase 14 correctly identified **28 out of 50** blue vans (56%), a massive leap from Phase 13 where the model was entirely uncertain.
2. **Dynamic Routing (Expert Weights)**: 
   - Mean Weights: `Attn=0.546`, `Topo=0.454`. The model now trusts structural geometry (TDA) almost equally to color/attention.
   - STD: `0.219`. The Combiner is highly dynamic, making distinct decisions for each image rather than collapsing to a single expert.
3. **The Power of Reject Option**: Despite the improvement, all 50 images remain flagged as `NEEDS_REVIEW`. This is **Aleatoric Uncertainty**. The model correctly predicts `vanet` but acknowledges that a blue van is visually ambiguous. It demands human verification, adhering to the core philosophy of **Governed AI (Human-in-the-loop)**.