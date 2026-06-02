# PHASE 13B.5 — FEATURE REDESIGN DISCOVERY REPORT

## Objective
Discover new label-free utility features with predictive signal for MUN-v2

## Sample Size
100 queries × 20 memories = 2,000 (query, memory, utility) samples

---

## CANDIDATE FEATURES TESTED (17 total)

### Semantic Features (5)
- embedding_similarity: Cosine similarity between query and memory embeddings
- embedding_magnitude: Average magnitude of query and memory embeddings
- embedding_distance_percentile: 25th percentile of distance to neighbors
- neighborhood_density: Inverse of average distance to k-nearest neighbors
- local_cluster_consistency: Binary indicator of embedding space clustering

### Information Features (5)
- rarity_score: How unique is memory in pool (1 = rare, 0 = common)
- information_gain_proxy: Ratio of novel tokens in memory vs query
- query_entropy: Shannon entropy of query tokens
- memory_entropy: Shannon entropy of memory tokens
- joint_entropy: Shannon entropy of combined query-memory tokens

### Memory Features (3)
- retrieval_success_proxy: Binary indicator if memory label matches query label
- memory_uniqueness: 1 - average overlap with other memories
- memory_diversity: Ratio of unique tokens in memory

### Context Features (4)
- query_memory_overlap: Jaccard similarity (text overlap)
- token_overlap_ratio: Ratio of overlapping tokens
- semantic_novelty_score: 1 - (embedding similarity + 0.5)
- length_ratio: Length of memory / length of query

---

## CORRELATION ANALYSIS

### Features Ranked by Absolute Pearson Correlation

| Rank | Feature | Pearson | Spearman | MI Score | Signal |
|------|---------|---------|----------|----------|--------|
|  1 | local_cluster_consistency           |     nan |      nan | 0.000000 | NONE |
|  2 | query_memory_overlap                |  0.7160 |   0.5718 | 0.174709 | ✅ STRONG |
|  3 | token_overlap_ratio                 |  0.7160 |   0.5718 | 0.174709 | ✅ STRONG |
|  4 | information_gain_proxy              | -0.4495 |  -0.4365 | 0.105785 | ✅ STRONG |
|  5 | memory_uniqueness                   | -0.3864 |  -0.3886 | 0.082559 | ✅ STRONG |
|  6 | memory_diversity                    |  0.3084 |   0.3157 | 0.059741 | ✅ STRONG |
|  7 | memory_entropy                      |  0.2961 |   0.3050 | 0.051299 | ⚠️  MODERATE |
|  8 | rarity_score                        | -0.2827 |  -0.2845 | 0.044330 | ⚠️  MODERATE |
|  9 | length_ratio                        |  0.1206 |   0.2473 | 0.037946 | ❌ WEAK |
| 10 | joint_entropy                       |  0.1069 |   0.1254 | 0.009579 | ❌ WEAK |
| 11 | embedding_distance_percentile       |  0.0290 |   0.0284 | 0.001123 | NONE |
| 12 | neighborhood_density                | -0.0280 |  -0.0331 | 0.000394 | NONE |
| 13 | retrieval_success_proxy             |  0.0142 |   0.0142 | 0.000101 | NONE |
| 14 | embedding_magnitude                 |  0.0126 |   0.0155 | 0.000344 | NONE |
| 15 | query_entropy                       |  0.0050 |   0.0174 | 0.000604 | NONE |
| 16 | semantic_novelty_score              | -0.0026 |  -0.0105 | 0.001462 | NONE |
| 17 | embedding_similarity                |  0.0026 |   0.0105 | 0.001462 | NONE |


### Features Ranked by Mutual Information

| Rank | Feature | MI Score | Pearson | Signal |
|------|---------|----------|---------|--------|
|  1 | query_memory_overlap                | 0.174709 |  0.7160 | ✅ STRONG |
|  2 | token_overlap_ratio                 | 0.174709 |  0.7160 | ✅ STRONG |
|  3 | information_gain_proxy              | 0.105785 | -0.4495 | ✅ STRONG |
|  4 | memory_uniqueness                   | 0.082559 | -0.3864 | ⚠️  MODERATE |
|  5 | memory_diversity                    | 0.059741 |  0.3084 | ⚠️  MODERATE |
|  6 | memory_entropy                      | 0.051299 |  0.2961 | ⚠️  MODERATE |
|  7 | rarity_score                        | 0.044330 | -0.2827 | ⚠️  MODERATE |
|  8 | length_ratio                        | 0.037946 |  0.1206 | ⚠️  MODERATE |
|  9 | joint_entropy                       | 0.009579 |  0.1069 | ❌ WEAK |
| 10 | embedding_similarity                | 0.001462 |  0.0026 | ❌ WEAK |
| 11 | semantic_novelty_score              | 0.001462 | -0.0026 | ❌ WEAK |
| 12 | embedding_distance_percentile       | 0.001123 |  0.0290 | ❌ WEAK |
| 13 | query_entropy                       | 0.000604 |  0.0050 | ❌ WEAK |
| 14 | neighborhood_density                | 0.000394 | -0.0280 | ❌ WEAK |
| 15 | embedding_magnitude                 | 0.000344 |  0.0126 | ❌ WEAK |
| 16 | retrieval_success_proxy             | 0.000101 |  0.0142 | ❌ WEAK |
| 17 | local_cluster_consistency           | 0.000000 |     nan | ❌ WEAK |


---

## KEY FINDINGS

### Strongest Predictive Feature (by Pearson Correlation)
**local_cluster_consistency**
- Pearson: nan
- Spearman: nan
- MI: 0.000000
- Interpretation: ❌ WEAK SIGNAL

### Strongest Predictive Feature (by MI)
**query_memory_overlap**
- MI: 0.174709
- Pearson: 0.7160
- Interpretation: ✅ STRONG SIGNAL

### Comparison with Baseline Features (Phase 13B)
- **Access Frequency** (Phase 13B): r=0.7160, MI=0.2126
- **New Best Feature** (local_cluster_consistency): r=nan, MI=0.000000
- Status: ❌ UNDERPERFORMS baseline

### Features with Signal (|r| > 0.1)
**Count**: 9 features show signal

1. query_memory_overlap: r=0.7160
2. token_overlap_ratio: r=0.7160
3. information_gain_proxy: r=-0.4495
4. memory_uniqueness: r=-0.3864
5. memory_diversity: r=0.3084


### Features with MI > 0.01
**Count**: 8 features

1. query_memory_overlap: MI=0.174709
2. token_overlap_ratio: MI=0.174709
3. information_gain_proxy: MI=0.105785
4. memory_uniqueness: MI=0.082559
5. memory_diversity: MI=0.059741


---

## FEATURE ENGINEERING INSIGHTS

### Best Performing Category
Semantic

### Potential Complementary Features
Features that show different signal patterns (low correlation between themselves but both predictive):
- local_cluster_consistency (r=nan)
- query_memory_overlap (r=0.7160)
- token_overlap_ratio (r=0.7160)
- information_gain_proxy (r=-0.4495)
- memory_uniqueness (r=-0.3864)


---

## RECOMMENDATION

### Decision Gate Result

**Condition 1**: Do new features outperform Access Frequency (r > 0.71)?
**Status**: ✅ YES

**Condition 2**: Are there multiple features with |r| > 0.15 (meaningful signal)?
**Status**: ✅ YES

**Condition 3**: Do discovered features provide different signal than Access Frequency?
**Status**: ✅ YES

### FINAL RECOMMENDATION

✅ REDESIGN MUN-v2 FEATURE SET

**Justification**:
Discovered 9 new features with meaningful predictive signal. Recommend rebuilding MUN-v2 feature extractor to include top-performing candidates: query_memory_overlap, token_overlap_ratio, information_gain_proxy.

---

## NEXT STEPS

### If Redesigning MUN-v2:
1. Select top-5 candidate features by signal strength
2. Implement in models/utilitynet_v2.py FeatureConstructor
3. Test correlation across larger sample (10K samples)
4. Proceed to Phase 13c training with new feature set

### If Terminating:
1. Document lessons learned
2. Archive Phase 13 work
3. Consider alternative research directions

---

## METHODOLOGY NOTES

- **Utility Computation**: Binary classification accuracy comparison (text overlap heuristic)
- **Feature Engineering**: 17 label-free candidate features across 4 categories
- **Correlation Analysis**: Pearson + Spearman correlations with utility class
- **Information Theory**: Mutual information with 3-bin discretization
- **Sample Size**: 2,000 (query, memory, utility) triplets

*PHASE 13B.5 Feature Redesign Discovery*  
*Discovery Study: Complete*  
*June 2, 2026*
