# VIDEX Phase 9.1 — Evaluation Correctness Audit & Mathematical Foundations

## 1. Executive Summary

This report establishes the mathematical correctness, algorithmic choices, and aggregation justifications for the VIDEX evaluation subsystem. It formally documents:
1. The difference between **evaluation harness verification** and **model accuracy benchmarking**.
2. Optimal bipartite matching vs greedy assignment, including an adversarial counterexample.
3. Micro-averaged vs macro-averaged metrics across multi-video corpora.
4. Division-by-zero safeguards and qualified denominator handling.
5. OCR/ASR normalization, Levenshtein edit distance, and mathematical conditions under which CER/WER exceed 1.0.
6. QA abstention confusion matrix and non-scoring of absent outputs.
7. Structural evidence integrity vs semantic ground truth.

---

## 2. Harness Verification vs Model Accuracy Benchmark

A critical distinction enforced in VIDEX Phase 9.1:
- **Harness Verification**: Verifies that ground-truth schemas, metric formulas, and evaluation runners execute deterministically and reject invalid inputs. Executing a harness without model predictions verifies ground-truth readiness, but produces **no accuracy benchmark**.
- **Model Accuracy Benchmark**: Requires valid, model-generated predictions adhering to `PredictionDataset`. Predictions are evaluated against gold-standard annotations to measure true empirical accuracy.

When the VIDEX CLI runs without predictions (`--predictions` omitted), it explicitly prints:
```text
[!] NOTICE: Evaluation executed WITHOUT model predictions file.
    Ground-truth readiness verified; no model accuracy benchmark claimed.
```
All modality accuracy metrics are marked `MetricStatus.UNAVAILABLE`, ensuring empty inputs never masquerade as zero accuracy or successful abstentions.

---

## 3. Bipartite Matching: Optimal vs Greedy

### 3.1 Mathematical Problem Definition

Let $G = \{g_1, \dots, g_N\}$ be the set of ground-truth detections, and $P = \{p_1, \dots, p_M\}$ be the set of model predictions. An edge $(g_i, p_j)$ is valid if:
1. $\text{class}(g_i) = \text{class}(p_j)$ (case-insensitive).
2. $|t(g_i) - t(p_j)| \le \Delta t_{\text{tol}}$ (temporal tolerance, default $0.10$s).
3. $\text{IoU}(g_i, p_j) \ge \theta_{\text{IoU}}$ (spatial overlap threshold, default $0.50$).

The objective is to find a matching $\mathcal{M} \subseteq G \times P$ such that:
1. Every $g_i \in G$ appears at most once in $\mathcal{M}$.
2. Every $p_j \in P$ appears at most once in $\mathcal{M}$.
3. **Primary objective**: Maximize the cardinality $|\mathcal{M}|$ (number of True Positives).
4. **Secondary objective**: Among matchings of maximum cardinality, maximize the total weight $\sum_{(g, p) \in \mathcal{M}} \text{IoU}(g, p)$.

### 3.2 The Adversarial Counterexample: Where Greedy Fails

Consider $N=2$ ground-truth boxes and $M=2$ predictions:
- Prediction $p_1$ overlaps $g_1$ with $\text{IoU} = 0.90$ and $g_2$ with $\text{IoU} = 0.80$.
- Prediction $p_2$ overlaps $g_1$ with $\text{IoU} = 0.70$ and $g_2$ with $\text{IoU} = 0.00$.

#### Greedy Assignment (Sorted by IoU Descending):
1. Candidate $(g_1, p_1, 0.90)$ is selected: $g_1 \leftrightarrow p_1$ matched.
2. Candidate $(g_2, p_1, 0.80)$ cannot match ($p_1$ already assigned).
3. Candidate $(g_1, p_2, 0.70)$ cannot match ($g_1$ already assigned).
**Greedy Outcome: $\text{TP} = 1$, $\text{FP} = 1$, $\text{FN} = 1$.**

#### Optimal Maximum-Weight Bipartite Assignment:
- Match $g_1 \leftrightarrow p_2$ ($\text{IoU} = 0.70$).
- Match $g_2 \leftrightarrow p_1$ ($\text{IoU} = 0.80$).
Both pairs satisfy $\text{IoU} \ge 0.50$.
**Optimal Outcome: $\text{TP} = 2$, $\text{FP} = 0$, $\text{FN} = 0$, Sum of IoUs = $1.50$.**

### 3.3 Implementation Details

VIDEX implements maximum-weight bipartite matching in pure Python using the Kuhn-Munkres (Hungarian) algorithm in $O(V^3)$ time (`min_cost_assignment`). To guarantee maximum cardinality first and sum of weights second, each valid edge receives:
$$w_{\text{effective}}(g_i, p_j) = 100.0 + \text{IoU}(g_i, p_j)$$
Any additional match contributes $\ge 100.0$, strictly dominating any fractional weight trade-offs.

Both `evaluate_detections` and `evaluate_temporal_events` support `matching_strategy="optimal"` (default) and `matching_strategy="greedy"`.

---

## 4. Micro-Averaged vs Macro-Averaged Aggregation

### 4.1 Micro-Averaging Across Videos

In multi-video evaluation, taking a weighted mean of per-video F1 scores is mathematically incorrect.

VIDEX computes true corpus-wide micro-averages by summing raw event counts across all videos first:
$$\text{Total TP} = \sum_{v \in V} \text{TP}_v, \quad \text{Total FP} = \sum_{v \in V} \text{FP}_v, \quad \text{Total FN} = \sum_{v \in V} \text{FN}_v$$

$$\text{Precision}_{\text{micro}} = \frac{\text{Total TP}}{\text{Total TP} + \text{Total FP}}$$

$$\text{Recall}_{\text{micro}} = \frac{\text{Total TP}}{\text{Total TP} + \text{Total FN}}$$

$$\text{F1}_{\text{micro}} = \frac{2 \cdot \text{Precision}_{\text{micro}} \cdot \text{Recall}_{\text{micro}}}{\text{Precision}_{\text{micro}} + \text{Recall}_{\text{micro}}}$$

### 4.2 Macro-Averaging Across Videos

Macro-averages are computed as unweighted means across videos where the per-video metric is defined:
$$\text{Precision}_{\text{macro}} = \frac{1}{|V_{\text{valid}}|} \sum_{v \in V_{\text{valid}}} \text{Precision}_v$$
Macro metrics are explicitly labeled with `macro` in both CLI tables and JSON outputs.

---

## 5. Division by Zero & Qualified Denominators

When a model predicts 0 objects for a video containing ground-truth items:
- $\text{TP} = 0, \text{FP} = 0, \text{FN} = N$.
- Denominator for precision ($\text{TP} + \text{FP} = 0$) is zero.

In VIDEX:
- Precision is set to $0.0000$ with explicit qualification in metadata:
  ```json
  "metadata": {
    "zero_denominator": true,
    "undefined_reason": "Zero positive predictions made (TP + FP = 0); precision is formally undefined"
  }
  ```
- The CLI summary renders: `0.0000 (undefined: 0/0)`.
- Undefined precision or recall never silently masquerades as an unqualified perfect score or generic zero.

---

## 6. Text Normalization, CER, and WER

### 6.1 Normalization Pipeline
1. **Unicode Decomposition**: `unicodedata.normalize("NFKC", text)` standardizes combining diacritics and ligatures (critical for Devanagari Hindi text).
2. **Casing**: Case-insensitive by default (`lower()`).
3. **Punctuation**: Stripped for ASR WER evaluation via Unicode category `P*`.
4. **Whitespace**: Multiple spaces, tabs, and newlines collapsed to a single ASCII space; leading and trailing whitespace stripped.

### 6.2 Error Rates Exceeding 1.0 (100%)

Character Error Rate (CER) and Word Error Rate (WER) are defined via Levenshtein edit distance:
$$\text{CER / WER} = \frac{S + D + I}{N_{\text{ref}}}$$
where $S$ is substitutions, $D$ is deletions, and $I$ is insertions.

> [!NOTE]
> If a model hallucinates text (e.g. Reference: `"cat"` ($N_{\text{ref}} = 3$), Hypothesis: `"catastrophe"` ($S=0, D=0, I=8$)), then:
> $$\text{CER} = \frac{0 + 0 + 8}{3} = 2.6667 \quad (266.67\%)$$
> This is mathematically correct standard behavior. VIDEX never truncates or caps CER/WER at 1.0.

---

## 7. QA Abstention Confusion Matrix

For Question-Answering, the system evaluates the binary decision to abstain when insufficient evidence exists:

| Metric Element | Condition | Interpretation |
|---|---|---|
| **True Positive ($\text{TP}_{\text{abstain}}$)** | `is_answerable=False`, `did_abstain=True` | Correct abstention on unanswerable question |
| **True Negative ($\text{TN}_{\text{answer}}$)** | `is_answerable=True`, `did_abstain=False` | Correct attempt on answerable question |
| **False Positive ($\text{FP}_{\text{refusal}}$)** | `is_answerable=True`, `did_abstain=True` | Unjustified refusal on answerable question |
| **False Negative ($\text{FN}_{\text{hallucination}}$)** | `is_answerable=False`, `did_abstain=False` | Hallucinated answer on unanswerable question |
| **Missing Output ($\text{Missing}$)** | Prediction omitted | Evaluated as missing output; **never scored as abstention** |

Metrics reported:
- **Decision Accuracy**: $\frac{\text{TP}_{\text{abstain}} + \text{TN}_{\text{answer}}}{\text{Total Evaluated}}$
- **Abstention Recall**: $\frac{\text{TP}_{\text{abstain}}}{\text{Unanswerable Total}}$
- **Abstention Precision**: $\frac{\text{TP}_{\text{abstain}}}{\text{TP}_{\text{abstain}} + \text{FP}_{\text{refusal}}}$
- **Abstention F1**: Harmonic mean of precision and recall.

---

## 8. Structural Evidence Integrity vs Semantic Entailment

`evaluate_evidence_reference_integrity` verifies:
1. Whether cited evidence IDs exist in the canonical `InMemoryGraphStore`.
2. Whether cited evidence IDs belong exclusively to the evaluated video namespace.

This metric is strictly **structural**. A claim referencing a valid bounding box in the correct video achieves 100% structural integrity even if the box does not semantically support the natural language claim. Semantic grounding requires multimodal QA verification against gold-standard QA annotations.
