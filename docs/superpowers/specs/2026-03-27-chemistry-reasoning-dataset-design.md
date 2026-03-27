# Chemistry Reasoning Dataset Design: LogD via GraphGen

**Date:** 2026-03-27
**Status:** Approved
**Goal:** Replace GNN-based ADMET property prediction workflows with a single LLM capable of both predicting and reasoning about molecular properties, starting with LogD.

---

## 1. Problem Statement

Current GNN pipelines produce accurate numeric predictions but no reasoning. The goal is to fine-tune Llama 3.2 (3B/8B) to:
1. Accept a SMILES string as input
2. Parse its own functional groups as part of a reasoning chain (no RDKit preprocessing at inference)
3. Output a structured chain-of-thought + numeric LogD prediction + confidence interval

Success is defined as: Spearman R and RMSE comparable to the GNN baseline, with human-readable reasoning chains.

---

## 2. Architecture: Hybrid Dual-KG Approach

Two knowledge graphs are built and run through GraphGen in parallel, each targeting different reasoning capabilities.

### KG1 — Functional Group Rule Graph

**Purpose:** Teach the model chemistry vocabulary and how individual functional groups affect LogD. Fully computationally derived — no manual literature encoding.

**Nodes (~150–200):**
- Functional groups identified via RDKit SMARTS patterns (carboxylic acid, piperazine, fluorine, aromatic ring, sulfonamide, amide, ester, etc.)
- Each node stores computed attributes:
  - Crippen LogP fragment contribution (RDKit `Chem.rdMolDescriptors.CalcCrippenDescriptors` on FG-containing fragment)
  - HBD/HBA counts (RDKit Lipinski descriptors)
  - TPSA contribution (RDKit `CalcTPSA`)
  - Estimated ionization state at pH 7.4 (dimorphite-DL or pkasolver)
  - Molecular weight contribution

**Edges (~400–600):**
- Computed delta edges: δLogD when adding FG to reference scaffold (computed on matched molecular pairs from ChEMBL)
- Interaction edges derived from computed descriptors: "electron-withdrawing (σp > 0.3)", "H-bond donor competes with lipophilic surface"
- Ionization edges: "ionized fraction > 99% at pH 7.4 (Henderson-Hasselbalch from estimated pKa)"
- Structural adjacency effects: computed via RDKit atom environment analysis

**Source:** Fully RDKit-derived + OpenBabel for 3D ionization states. No manual literature encoding.

**Toolchain:**
- `RDKit` — SMARTS-based FG enumeration, Crippen fragments, Lipinski descriptors, TPSA, matched molecular pairs
- `dimorphite-DL` or `pkasolver` — pKa estimation and ionization state at pH 7.4
- `OpenBabel` — 3D conformer generation, additional physicochemical properties

**GraphGen modes used:**
- `atomic` → single-FG factual QA (Alpaca format)
- `cot` → multi-FG reasoning chains using Leiden community detection (ChatML/ShareGPT)

---

### KG2 — Molecule Graph

**Purpose:** Ground the model in real experimental data; teach SAR patterns and comparative reasoning.

**Nodes (~1,000 for POC):**
- Individual molecules with attributes: SMILES, experimental LogD, scaffold (Murcko), functional group list (RDKit-parsed at KG build time), LogD bin (low/mid/high)

**Edges (~3,000–5,000):**
- Tanimoto similarity > 0.6 (Morgan fingerprints, radius 2)
- Shared Murcko scaffold
- Same functional group class (e.g., both contain piperazine)
- Adjacent LogD bins (for gradient reasoning)

**Source:** ChEMBL experimental dataset (~1K for POC, ~60K for full scale).

**GraphGen modes used:**
- `multi_hop` → comparative reasoning: "why does compound A have higher LogD than B?"
- `aggregated` → scaffold-class patterns: "what LogD range do piperazine-containing compounds cluster in?"

---

## 3. Four Dataset Types

### SFT-1: Atomic FG Facts (Alpaca format)
- **Source:** KG1 via `atomic` mode
- **Volume:** ~2,000 examples
- **Format:**
  ```json
  {
    "instruction": "What effect does a carboxylic acid group have on LogD at pH 7.4?",
    "output": "Carboxylic acids have a pKa of approximately 4–5, meaning they are fully ionized (negatively charged) at physiological pH 7.4. This ionization dramatically increases aqueous solubility and decreases lipophilicity. A single carboxylic acid group typically reduces LogD by 1.5–2.5 units relative to the equivalent ester or methyl group."
  }
  ```
- **Purpose:** Build chemistry vocabulary before full prediction tasks.

### SFT-2: Full Prediction CoT (ChatML format)
- **Source:** KG1 (CoT mode) + KG2 nodes (real molecules)
- **Volume:** ~3,000 examples
- **Format:**
  ```
  <|user|>
  Predict the LogD at pH 7.4 for the following molecule: [SMILES]
  <|assistant|>
  Step 1 — Functional group identification:
  Examining the SMILES, I can identify: [list of FGs with positions]

  Step 2 — Lipophilic contributions:
  [FG-by-FG analysis with estimated unit contributions]

  Step 3 — Ionizable groups at pH 7.4:
  [pKa analysis, ionization state, correction]

  Step 4 — Combining contributions:
  [Summation with interaction effects noted]

  Prediction: LogD ≈ X.X (range: Y.Y – Z.Z, confidence: medium/high)
  ```
- **Purpose:** Core prediction capability with step-by-step reasoning.

### SFT-3: Comparative SAR (ChatML format)
- **Source:** KG2 via `multi_hop` and `aggregated` modes
- **Volume:** ~2,000 examples
- **Format:**
  ```
  <|user|>
  Compound A: [SMILES_A] (LogD = 2.1)
  Compound B: [SMILES_B] (LogD = 3.4)
  Explain the structural basis for the difference in LogD.
  <|assistant|>
  Comparing the two structures: Compound B contains an additional fluorine substituent on the aromatic ring...
  ```
- **Purpose:** Teach structure-activity relationship reasoning, not just single-molecule prediction.

### DPO: Preference Pairs (ChatML format)
- **Source:** Both KGs + GNN oracle
- **Volume:** ~2,000 pairs
- **GNN oracle:** The existing trained GNN from your current ADMET pipeline — no new model to build. Used purely as a "wrong number generator."
- **Generation pipeline:**
  1. For each molecule in KG2, run the existing GNN to get prediction `G`
  2. Run the base Llama model (zero-shot) to generate a reasoning chain + prediction `L`
  3. Compute `|L - experimental|` and `|G - experimental|`
  4. If `|L - experimental| < threshold` (e.g., 0.5 log units): chain is **chosen**
  5. If GNN prediction `G` diverges from experimental by > threshold: use `G` as the "wrong answer" target, prompt Llama to generate reasoning that arrives at `G` → **rejected**
  6. Additionally: systematically perturb correct chains (ignore ionization, miss an FG, apply wrong pH) → additional **rejected** examples
- **Format:** Standard DPO `{prompt, chosen, rejected}` triples
- **Purpose:** Align the model to reject chemically flawed reasoning even when it arrives at a plausible-sounding number.

---

## 4. Fine-Tuning Curriculum

Training proceeds in three stages on Llama 3.2 3B (and optionally 8B in parallel):

| Stage | Data | Method | Purpose |
|-------|------|---------|---------|
| 1 | SFT-1 (Atomic facts) | QLoRA SFT | Chemistry vocabulary — what each FG does |
| 2 | SFT-2 + SFT-3 (CoT + Comparative) | QLoRA SFT | Combine FG effects into predictions |
| 3 | DPO pairs | DPO (β=0.1) | Reject flawed reasoning, align to experimental values |

**Rationale for curriculum:** A 3B model given full prediction CoT before it knows FG semantics will hallucinate chemistry. Stage 1 acts as vocabulary pretraining for the domain.

---

## 5. KG Construction

### KG1 Build Process
1. Define ~150–200 functional groups as SMARTS patterns (`fg_smarts.yaml`)
2. For each FG, enumerate a set of representative fragment molecules using RDKit
3. Compute node attributes per FG: Crippen LogP contribution, HBD/HBA, TPSA, MW contribution
4. Estimate pKa and ionization fraction at pH 7.4 using dimorphite-DL (or pkasolver)
5. Build delta edges using matched molecular pairs from ChEMBL: find pairs differing by exactly one FG, compute δLogD from experimental values
6. Add interaction edges from computed descriptor correlations (e.g., electron-withdrawing σp values from RDKit)
7. Export as GraphML using NetworkX → feed into GraphGen via `GraphmlReader`

**Key script:** `chemistry/kg1_build/build_kg1.py` — fully automated, no manual curation

### KG2 Build Process
1. Load ChEMBL experimental dataset (SMILES + LogD)
2. Parse functional groups with RDKit (at KG build time only — not at inference)
3. Compute Morgan fingerprints (radius=2, 2048 bits) for all molecules
4. Build Tanimoto similarity edges (threshold 0.6) using RDKit `BulkTanimotoSimilarity`
5. Compute Murcko scaffolds, add scaffold-sharing edges
6. Export as GraphML → feed into GraphGen

---

## 6. GraphGen Configuration

Each KG uses a separate GraphGen config file:

**KG1 configs:**
- `chemistry_atomic_config.yaml` — atomic mode, Alpaca output, ~2K target
- `chemistry_cot_config.yaml` — CoT mode with Leiden partitioning, ChatML output, ~3K target

**KG2 configs:**
- `chemistry_multihop_config.yaml` — multi-hop mode, ECE partitioning, ChatML output, ~2K target (→ SFT-3)
- `chemistry_aggregated_config.yaml` — aggregated mode, ChatML output, ~1K target (→ SFT-2, anchoring CoT with real molecule values)

**Synthesizer model:** Claude Sonnet 4 (via LiteLLM/Bedrock) for data generation quality
**Trainee model:** Llama 3.2 3B (via LiteLLM/Bedrock) for ECE-based knowledge gap targeting

---

## 7. Evaluation

| Metric | GNN Baseline | Target for POC |
|--------|-------------|----------------|
| Spearman R | ~0.76 (Claude 3.7 ZS) | ≥ 0.75 |
| RMSE | ~2.1 log units | ≤ 2.0 log units |
| Reasoning quality | N/A | Manual review of 50 chains |
| FG identification accuracy | N/A | ≥ 80% of FGs correctly identified |

Test set: held-out 10% of ChEMBL data (not used in KG2 construction).

---

## 8. POC Scope & Scale Path

**POC (this spec):**
- KG1: ~150 nodes, ~400 edges
- KG2: ~1,000 molecules
- Total training examples: ~9,000
- Model: Llama 3.2 3B with QLoRA

**Scale path (post-POC):**
- KG2: Expand to ~60K ChEMBL molecules
- Add ADMET properties: hERG, Clint, VDSS as additional KG1 rule subgraphs
- DPO pool grows proportionally with real data
- Larger model: Llama 3.1 70B or fine-tune via Bedrock

---

## 9. File Structure

```
GraphGen/
├── chemistry/
│   ├── kg1_build/
│   │   ├── fg_smarts.yaml                # SMARTS patterns for ~150 FGs
│   │   ├── build_kg1.py                  # RDKit/OpenBabel → GraphML (fully automated)
│   │   └── chemistry_rule_graph.graphml  # KG1 output
│   ├── kg2_build/
│   │   ├── build_kg2.py                  # ChEMBL → molecule GraphML (RDKit edges)
│   │   └── molecule_graph.graphml        # KG2 output
│   ├── configs/
│   │   ├── chemistry_atomic_config.yaml
│   │   ├── chemistry_cot_config.yaml
│   │   ├── chemistry_multihop_config.yaml
│   │   └── chemistry_aggregated_config.yaml
│   ├── dpo/
│   │   └── generate_dpo_pairs.py         # GNN oracle + DPO pair builder
│   └── evaluate/
│       └── evaluate_logd.py              # Spearman R, RMSE, chain review
├── requirements-chemistry.txt            # rdkit, openbabel, dimorphite-dl, pkasolver
```
