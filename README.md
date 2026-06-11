# Predicting Band Gaps of DRAM High-k Dielectric Candidates
**ML-based material screening with inorganic chemistry interpretation**

---

## Background & Motivation

DRAM scaling requires gate oxides with **high dielectric constant (high-k) AND wide band gap** simultaneously.
The standard SiO₂ (k≈3.9, Eg≈9 eV) cannot meet both requirements as nodes shrink — finding replacement materials is a critical challenge.

This project uses machine learning to **screen inorganic oxide candidates** and interprets the model's decisions through inorganic chemistry principles (crystal field theory, d-orbital occupancy), going beyond simple accuracy metrics.

---

## Project Pipeline

```
Materials Project API
        ↓
  153,328 materials (band gap 0–6 eV)
        ↓
  Filter: stable oxides, Eg > 3 eV, no toxic elements
        ↓
  3,914 high-k candidates
        ↓
  Magpie feature extraction (132 features)
  [electronegativity, d-orbital occupancy, atomic radius, ...]
        ↓
  XGBoost regression
        ↓
  MAE: 0.287 eV  |  R²: 0.749
        ↓
  Domain interpretation (inorganic chemistry)
```

---

## Key Results

### Model Performance
| Metric | Value |
|--------|-------|
| MAE | **0.287 eV** |
| R² | **0.749** |
| Training set | 3,131 materials |
| Test set | 783 materials |

> Composition-only features (no crystal structure) — competitive with literature benchmarks (~0.7–0.85 R²)

### Known high-k materials — model sanity check
| Material | Actual Eg | Predicted | Use |
|----------|-----------|-----------|-----|
| HfO₂ | 4.02 eV | ✓ | Current DRAM gate oxide standard |
| ZrO₂ | 3.53 eV | ✓ | HfO₂ replacement candidate |
| Al₂O₃ | 5.87 eV | ✓ | Interfacial layer |
| TiO₂ | 2.06 eV | filtered out | High k but low Eg → leakage current issue |

---

## Domain Interpretation

### Why NdValence accounts for 44% of feature importance

The top 4 features are all variants of **NdValence** (d-orbital electron count):

| Feature | Importance | Physical meaning |
|---------|-----------|-----------------|
| avg_dev NdValence | 24.5% | d-electron diversity across constituent elements |
| maximum NdValence | 10.4% | highest d-electron count in composition |
| range NdValence | 5.8% | d-orbital occupancy spread |
| mean NdValence | 3.3% | average d-electron count |

**Crystal field theory explanation:**  
In d⁰ transition metal oxides (Hf⁴⁺, Zr⁴⁺, Al³⁺), the d orbitals are completely empty.  
The energy gap between the O 2p valence band and the empty metal d conduction band is maximized → **wide band gap**.  
When d electrons are present (V⁴⁺, Fe³⁺), they introduce mid-gap states or lower the conduction band → **band gap decreases**.

**Why avg_dev ranks higher than mean:**  
Mixed-cation oxides containing both d⁰ and dⁿ cations create charge transfer pathways between sites → gap reduction. The *diversity* of d-orbital filling, not just the average, drives this effect.

---

### Electronegativity: why linear correlation is weak (r = 0.078)

Inorganic chemistry predicts: higher electronegativity difference → stronger ionic bonding → wider band gap.  
This holds qualitatively (Hf-O more ionic than Ti-O → HfO₂ > TiO₂ gap),  
but the **mean electronegativity across the whole composition is dominated by O (EN = 3.44)**, masking the cation signal.

→ Implication: separating cation-only electronegativity as a feature would improve the model.

---

### Prediction error analysis — when hypotheses meet data

**Initial hypothesis:** Lanthanide compounds (4f orbitals) would show high error because Magpie's NdValence doesn't capture f electrons.

**Actual result:**

| Group | MAE |
|-------|-----|
| H-containing (hydrates) | **0.381 eV** (+41%) |
| No H | 0.270 eV |
| Lanthanide-containing | **0.237 eV** (better than average) |
| No lanthanides | 0.307 eV |

**Revised interpretation:**  
Lanthanide oxides (La₂O₃, Y₂O₃, etc.) follow consistent patterns and are well-represented in training data — the model learned them well.  
The actual problem is **H-containing complex oxides**: O–H bonding fundamentally alters the electronic structure in ways the Magpie d-orbital descriptors cannot capture.

→ Improvement path: H-specific bonding descriptors

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Data source | Materials Project API (`mp-api`) |
| Feature engineering | `matminer` (Magpie preset) |
| ML model | `XGBoost` |
| Data processing | `pandas`, `numpy` |
| Visualization | `matplotlib` |
| Environment | Python 3.14, VS Code |

---

## Notebooks

| Notebook | Content |
|----------|---------|
| `01_data_collection.ipynb` | Materials Project API query, 153k materials |
| `02_EDA.ipynb` | Band gap distribution, high-k candidate filtering |
| `03_feature_engineering.ipynb` | Magpie feature extraction (132 features) |
| `04_model_training.ipynb` | XGBoost training, performance evaluation |
| `05_interpretation.ipynb` | **Domain interpretation — the core differentiator** |

---

## Future Work

- [ ] Add cation-only electronegativity as explicit feature
- [ ] Include crystal structure features (space group, coordination number) for better accuracy
- [ ] Expand to dielectric constant (k) prediction — dual objective screening (high k + wide Eg)
- [ ] Phase 2: CVD/ALD process parameter → thin film property prediction

---

## Author

Materials scientist & engineer with PhD in inorganic chemistry (POSTECH).  
Research background: metal nanocluster synthesis, 2D MOF electrocatalysts, oxide thin film processing.

> *"Accuracy numbers tell you how well a model works. Domain interpretation tells you why — and what to fix next."*
