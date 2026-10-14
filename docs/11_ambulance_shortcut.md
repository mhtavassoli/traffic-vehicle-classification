## Phase 11 Plan: Multi-Region Attention + Fuzzy Head

### Architecture
Input → ResNet18 → Feature Map (B, 512, 4, 4)
                     ↓
              Split into 3 regions:
              - Top (upper body/lights)
              - Middle (text/body)
              - Bottom (license plate/wheels)
                     ↓
              MultiRegionAttention (concatenate + Linear)
                     ↓
              FuzzyHead (Gaussian membership, 8 classes)
                     ↓
              Calibrated output (temperature-scaled)

### Hypothesis
- Multi-region attention prevents shortcut on lower body.
- Fuzzy head allows soft decisions (ambulance OR vanet) with confidence.
- Expected result: test3 blue vans → vanet with high confidence.