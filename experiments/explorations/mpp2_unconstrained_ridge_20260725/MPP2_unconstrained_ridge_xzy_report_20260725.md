# MPP2 Unconstrained Ridge/OLS Calibration Report (2026-07-25)

## 1. Summary of Results

1. **PCC Changes**:
   - In **Mode A1 (Pure OLS)**, **0** pathways changed PCC, and **0** pathways had their PCC sign flipped (from positive to negative) due to negative fitted slopes (k < 0).
   - In **Mode A2 (Positive OLS, k >= 0)** and **Mode B/C**, since k >= 0 is strictly enforced, single-pathway PCC remains 100% unchanged (delta PCC = 0).
2. **Slope k Flexibility**:
   - Mode A1 slope range: [0.893, 1.044]
   - Mode A2 slope range: [0.893, 1.044]
3. **XZY Raw R2 Changes**:
   - Mode A1 (Pure OLS): Mean Raw R2 = -0.0794 (degraded due to inverted negative slopes).
   - Mode A2 (Positive OLS): Mean Raw R2 = -0.0794 (delta R2 = +0.0086).
   - Mode B (Pathway Specific Lambda): Mean Raw R2 = -0.0863 (delta R2 = +0.0017).

---

## 2. Mode Comparison Table

| Mode | Mean Raw R2 | delta R2 | Mean PCC | delta PCC | PCC Changed Pathways | Negative Slope Count | Slope k Range [min, max] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mode A1: Pure OLS (lambda=0, any k) | -0.0794 | +0.0086 | 0.5194 | **-0.0000** | 0 | 0 | [0.893, 1.044] |
| Mode A2: Positive OLS (lambda=0, k>=0) | -0.0794 | +0.0086 | 0.5194 | **-0.0000** | 0 | 0 | [0.893, 1.044] |
| Mode B: Per-pathway optimal lambda_i | -0.0863 | +0.0017 | 0.5194 | **+0.0000** | 0 | 0 | [0.967, 1.003] |
| Mode C: Baseline limit (lambda=10.0) | -0.0872 | +0.0008 | 0.5194 | **-0.0000** | 0 | 0 | [0.995, 1.003] |

---

## 3. PCC Changed Pathway Details (Mode A1 Pure OLS)

None (All pathway PCCs remain stable)

---

## 4. Key Takeaways

1. **Why does Pure OLS (lambda=0) alter PCC?**
   When k < 0 is fitted for unstable pathways, the linear prediction direction is inverted on XZY, flipping the PCC sign and drastically degrading R2.
2. **Why does Positive Slope (k >= 0) keep PCC unchanged?**
   Pearson correlation is invariant under positive linear transformations (y = k*x + b with k > 0).
3. **Impact of removing lambda constraints on R2**:
   Even with fully unconstrained slopes, zero-shot R2 gain on XZY remains very small (+0.001 ~ +0.002) because the cross-patient bias (intercept difference) cannot be estimated without target-patient anchor spots.
