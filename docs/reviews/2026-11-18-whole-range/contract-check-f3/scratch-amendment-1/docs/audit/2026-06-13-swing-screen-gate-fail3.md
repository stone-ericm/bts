# Swing campaign Stage-1 screen report (amendment #2) — 2026-06-13

Seeds: [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]; screen days: 88; primary stat: paired daily rank-AUC delta (seed-averaged), week-blocked sign-permutation p

## GATES — FAILED (stop; do not read families as signal)
- GATE 1 gross sentinel: FAIL — delta -0.00820 (every seed positive: False; >=3x null band 0.00456: False)
- GATE 2 M3 sentinel: FAIL — delta -0.00045 vs null band 0.00456 (p=0.723)
- GATE 3 nulls unremarkable: FAIL ({'ctl_mask_only': 0.00037, 'ctl_permuted': 0.00456})

Empirical practical-null band (max |null arm delta|): 0.00456

## Families (primary stat; > null band AND p<0.05 marked ***)
### P
- `p_miss_30g`: aucΔ +0.00051 (p=0.087, seeds [0.00164, 0.0002, -0.00029, 0.00174, 0.00048, -0.00012, -0.00049, 0.00087, 0.00063, 0.0004]), ndcgΔ -0.00172, top1Δ +0.0034
- `p_miss_15g`: aucΔ +0.00038 (p=0.266, seeds [0.00088, -0.00071, -2e-05, 0.00093, 0.00065, -0.00042, 0.00094, 0.00083, 0.00038, 0.00032]), ndcgΔ -0.00520, top1Δ -0.0023
- `p_miss_std_30g`: aucΔ +0.00017 (p=0.283, seeds [0.00072, 0.00041, -0.0005, 0.0001, -0.00016, -0.0002, -0.00024, -0.00025, 0.00019, 0.00161]), ndcgΔ -0.00115, top1Δ +0.0023
- `p_miss_7g`: aucΔ -0.00031 (p=0.452, seeds [-0.00033, -0.0015, -0.00027, 0.00019, 9e-05, -0.0013, 0.00044, 0.00056, -0.00078, -0.0002]), ndcgΔ -0.00472, top1Δ -0.0170
- `p_miss_60g`: aucΔ -0.00050 (p=0.467, seeds [-0.00028, -0.00036, -0.00085, -0.00071, -0.00075, -0.00148, -0.00087, -0.0002, -0.00055, 0.00108]), ndcgΔ -0.00193, top1Δ +0.0114
- `p_high_share_30g`: aucΔ -0.00053 (p=0.483, seeds [6e-05, -8e-05, -0.00116, -0.00044, -0.00065, -0.0011, -0.00104, -0.00049, -0.00077, 0.00036]), ndcgΔ -0.00059, top1Δ +0.0216
- **verdict: alive; best `p_miss_30g`**

### B
- `b_miss_15g`: aucΔ -0.00009 (p=0.351, seeds [0.00048, 0.00051, -0.00011, -0.00081, -0.00017, -0.00078, -2e-05, 0.00045, -0.00032, -0.00013]), ndcgΔ -0.00202, top1Δ +0.0159
- `b_miss_std_30g`: aucΔ -0.00015 (p=0.742, seeds [0.00141, -0.00014, -7e-05, -0.00016, -0.00059, -0.00096, -0.00049, -0.00021, 0.00016, -0.00046]), ndcgΔ +0.00238, top1Δ +0.0102
- `b_miss_30g`: aucΔ -0.00040 (p=0.647, seeds [-0.00077, -0.00054, 0.00031, -0.00032, 0.00026, -0.00167, 0.00061, -0.00129, -0.00044, -0.00017]), ndcgΔ +0.00416, top1Δ +0.0125
- `b_miss_60g`: aucΔ -0.00059 (p=0.499, seeds [-0.00011, -0.00151, -0.00154, -0.00039, -0.00164, -0.0006, -0.00058, -0.00043, 7e-05, 0.00079]), ndcgΔ +0.00096, top1Δ +0.0136
- `b_miss_7g`: aucΔ -0.00089 (p=0.503, seeds [-0.00124, -0.00159, -0.00099, -0.00115, -0.00106, -0.00091, -0.0005, -0.00088, -0.00053, -8e-05]), ndcgΔ -0.00415, top1Δ -0.0170
- **verdict: alive; best `b_miss_15g`**

### T
- `t_miss_drift`: aucΔ -0.00031 (p=0.388, seeds [0.00102, -0.00175, -0.00049, 0.00012, -0.00076, -0.00011, -0.00067, 0.0002, -0.0003, -0.00038]), ndcgΔ -0.00199, top1Δ +0.0114
- `t_intercept_drift`: aucΔ -0.00036 (p=0.776, seeds [0.00026, -0.00092, -0.00119, 0.00027, 5e-05, -0.00091, -0.0003, 0.00035, -0.00067, -0.00056]), ndcgΔ -0.00059, top1Δ +0.0250
- **verdict: DEAD (consistently negative); best `t_miss_drift`**

### S
- `s_swinglen_drift`: aucΔ +0.00065 (p=0.067, seeds [0.00145, 0.00037, 9e-05, 0.00187, 0.0012, -0.00048, 0.00029, 0.00032, 0.00101, 0.00037]), ndcgΔ -0.00186, top1Δ +0.0000
- `s_attack_angle_30g`: aucΔ +0.00033 (p=0.256, seeds [0.00101, -0.00109, -0.00088, 0.00076, 0.00038, -3e-05, -0.00012, 0.00132, 0.00112, 0.00085]), ndcgΔ +0.01130, top1Δ +0.0205
- `s_attack_std_30g`: aucΔ -0.00151 (p=1.000, seeds [-2e-05, -0.00169, -0.00183, -0.00147, -0.00181, -0.00227, -0.00147, -0.00081, -0.00205, -0.00167]), ndcgΔ -0.00008, top1Δ -0.0080
- **verdict: alive; best `s_swinglen_drift`**

### M
- `m_high_alignment`: aucΔ +0.00014 (p=0.436, seeds [0.00082, 0.00078, -0.00066, -0.00074, 0.00076, -0.00059, -0.00011, 3e-05, -0.00013, 0.00126]), ndcgΔ +0.00730, top1Δ +0.0341
- `m_high_mismatch`: aucΔ -0.00068 (p=0.884, seeds [-0.00025, -0.00167, -0.00126, -0.00033, -0.0, -0.00163, -0.0005, -0.00032, -0.00101, 0.00013]), ndcgΔ +0.00776, top1Δ +0.0182
- **verdict: alive; best `m_high_alignment`**

## Omnibus
- `omni_P`: aucΔ -0.00030 (p=0.440), ndcgΔ -0.00540
- `omni_M`: aucΔ -0.00043 (p=0.713), ndcgΔ +0.00755
- `omni_S`: aucΔ -0.00050 (p=0.465), ndcgΔ +0.01088
- `omni_T`: aucΔ -0.00091 (p=0.473), ndcgΔ -0.00814
- `omni_B`: aucΔ -0.00169 (p=0.506), ndcgΔ -0.00838
- `omni_ALL`: aucΔ -0.00300 (p=0.975), ndcgΔ -0.00326

## PROPOSED FROZEN BUNDLE (pending human review): ['p_miss_30g', 'b_miss_15g', 's_swinglen_drift', 'm_high_alignment']

**GATES FAILED — bundle proposal void; family lines are diagnostics only.**