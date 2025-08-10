#!/usr/bin/env python3
"""
Analysis of whether TUFNWGTP weight scaling affects results.
"""

# Your actual values
nll_per_weight = 0.45970294467975964
top1_acc_weighted = 0.9020924428424754
total_weight = 54976006130056.086

print("=== WEIGHT SCALING ANALYSIS ===")
print(f"Current results:")
print(f"  nll_per_weight: {nll_per_weight}")
print(f"  top1_acc_weighted: {top1_acc_weighted}")
print(f"  total_weight: {total_weight:,.0f}")
print()

# Calculate what the original totals must have been
implied_total_nll = nll_per_weight * total_weight
implied_top1_weight = top1_acc_weighted * total_weight

print(f"Implied totals:")
print(f"  total_nll: {implied_total_nll:,.0f}")
print(f"  top1_weight: {implied_top1_weight:,.0f}")
print()

# If weights were divided by 1000
scaled_total_weight = total_weight / 1000
scaled_total_nll = implied_total_nll / 1000  # NLL scales proportionally with weights
scaled_top1_weight = implied_top1_weight / 1000

# Calculate scaled metrics
scaled_nll_per_weight = scaled_total_nll / scaled_total_weight
scaled_top1_acc = scaled_top1_weight / scaled_total_weight

print("=== IF WEIGHTS WERE DIVIDED BY 1000 ===")
print(f"Scaled totals:")
print(f"  scaled_total_weight: {scaled_total_weight:,.0f}")
print(f"  scaled_total_nll: {scaled_total_nll:,.0f}")
print(f"  scaled_top1_weight: {scaled_top1_weight:,.0f}")
print()

print(f"Scaled metrics:")
print(f"  nll_per_weight: {scaled_nll_per_weight}")
print(f"  top1_acc_weighted: {scaled_top1_acc}")
print()

print("=== COMPARISON ===")
print(f"NLL per weight:")
print(f"  Original: {nll_per_weight:.15f}")
print(f"  Scaled:   {scaled_nll_per_weight:.15f}")
print(f"  Difference: {abs(nll_per_weight - scaled_nll_per_weight):.2e}")
print()

print(f"Top-1 accuracy:")
print(f"  Original: {top1_acc_weighted:.15f}")
print(f"  Scaled:   {scaled_top1_acc:.15f}")
print(f"  Difference: {abs(top1_acc_weighted - scaled_top1_acc):.2e}")
print()

print("=== CONCLUSION ===")
if abs(nll_per_weight - scaled_nll_per_weight) < 1e-10 and abs(top1_acc_weighted - scaled_top1_acc) < 1e-10:
    print("✅ Weight scaling does NOT affect the final metrics!")
    print("   The ratios remain the same regardless of weight scale.")
else:
    print("❌ Weight scaling DOES affect the final metrics!")
    print("   This would indicate a problem in the calculation.")

print()
print("EXPLANATION:")
print("In the NLL calculation:")
print("  total_nll += -log(p) * w")
print("  nll_per_weight = total_nll / total_w")
print()
print("If all weights are scaled by factor k:")
print("  total_nll_scaled = k * total_nll")
print("  total_w_scaled = k * total_w")
print("  nll_per_weight_scaled = (k * total_nll) / (k * total_w) = total_nll / total_w")
print()
print("Similarly for accuracy:")
print("  top1_w_scaled = k * top1_w")
print("  acc_scaled = (k * top1_w) / (k * total_w) = top1_w / total_w")
print()
print("Therefore, uniform weight scaling should NOT affect normalized metrics.")
