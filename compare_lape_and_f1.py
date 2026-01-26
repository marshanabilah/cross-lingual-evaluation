import json
import numpy as np
import pandas as pd

# ===================================================================
# 1. LOAD DATA
# ===================================================================

print("Loading data...")

# lape_id_to_en = json.load(open('results/llama-3.2_ROME_id_lape_analysis.json'))
lape_en_to_id = json.load(open('results/llama-3.2_ROME_en_lape_analysis.json'))
# bizsre_id_to_en = json.load(open('results_lape/llama-3.2_ROME_id_en_per_edit_f1.json'))
bizsre_en_to_id = json.load(open('results_lape/llama-3.2_ROME_en_id_per_edit_f1.json'))

print("✓ Data loaded")

# ===================================================================
# 2. EXTRACT DATA
# ===================================================================

def extract_data(lape_results, bizsre_results):
    per_edit_lape = lape_results['per_edit_analysis']
    results = []
    
    for i in range(len(per_edit_lape)):
        lape = per_edit_lape[i]
        
        # Extract F1 - ADJUST THIS based on your data structure
        if isinstance(bizsre_results, list):
            f1 = bizsre_results[i]
        else:
            f1 = bizsre_results  # Adjust as needed
        
        # if direction == 'id_to_en':
        #     results.append({
        #         'edit_idx': i,
        #         'subject': lape.get('subject', f'edit_{i}'),
        #         'target_f1': f1.get('generalization_en_f1', 0),
        #         'lang_specific_pct': lape['overlap_low_lape_src_pct'] + lape['overlap_low_lape_tgt_pct'],
        #         'src_specific_pct': lape['overlap_low_lape_src_pct'],
        #         'tgt_specific_pct': lape['overlap_low_lape_tgt_pct'],
        #         'direction': 'id→en'
        #     })
        # else:
        results.append({
            'edit_idx': i,
            'subject': lape.get('subject', f'edit_{i}'),
            'target_f1': f1.get('generalization_id_f1', 0),
            'lang_specific_pct': lape['overlap_low_lape_src_pct'] + lape['overlap_low_lape_tgt_pct'],
            'src_specific_pct': lape['overlap_low_lape_src_pct'],
            'tgt_specific_pct': lape['overlap_low_lape_tgt_pct']
        })
    
    return results

# data = extract_data(lape_id_to_en, bizsre_id_to_en)
data = extract_data(lape_en_to_id, bizsre_en_to_id)

df = pd.DataFrame(data)
print(f"✓ Extracted {len(df)} edits\n")

# ===================================================================
# 3. GROUP BY QUARTILES
# ===================================================================

print("="*70)
print("GROUPING BY PERFORMANCE")
print("="*70)

q1 = df['target_f1'].quantile(0.25)
q3 = df['target_f1'].quantile(0.75)

print(f"\nF1 Quartiles:")
print(f"  Q1 (25%): {q1:.2f}")
print(f"  Q3 (75%): {q3:.2f}")

top_quartile = df[df['target_f1'] >= q3]
bottom_quartile = df[df['target_f1'] <= q1]

print(f"\nGroups:")
print(f"  Top 25% (F1 ≥ {q3:.2f}): {len(top_quartile)} edits")
print(f"  Bottom 25% (F1 ≤ {q1:.2f}): {len(bottom_quartile)} edits")

# ===================================================================
# 4. COMPARE
# ===================================================================

print("\n" + "="*70)
print("COMPARISON: LANGUAGE-SPECIFIC NEURONS")
print("="*70)

print(f"\nTop Performers:")
print(f"  Mean F1: {top_quartile['target_f1'].mean():.2f}")
print(f"  Mean lang-specific: {top_quartile['lang_specific_pct'].mean():.3f}%")

print(f"\nBottom Performers:")
print(f"  Mean F1: {bottom_quartile['target_f1'].mean():.2f}")
print(f"  Mean lang-specific: {bottom_quartile['lang_specific_pct'].mean():.3f}%")

print("\n1. Total language-specific (src + tgt):")
print(f"  Top: {top_quartile['lang_specific_pct'].mean():.3f}%")
print(f"  Bottom: {bottom_quartile['lang_specific_pct'].mean():.3f}%")

print("\n2. Source-specific only:")
print(f"  Top: {top_quartile['src_specific_pct'].mean():.3f}%")
print(f"  Bottom: {bottom_quartile['src_specific_pct'].mean():.3f}%")

print("\n3. Target-specific only:")
print(f"  Top: {top_quartile['tgt_specific_pct'].mean():.3f}%")
print(f"  Bottom: {bottom_quartile['tgt_specific_pct'].mean():.3f}%")

diff = bottom_quartile['lang_specific_pct'].mean() - top_quartile['lang_specific_pct'].mean()
print(f"\nDifference: {diff:.3f}%")

if diff > 0.1:
    print("→ Bottom performers use MORE language-specific neurons")
elif diff < -0.1:
    print("→ Top performers use MORE language-specific neurons")
else:
    print("→ Similar language-specific neuron usage")

# ===================================================================
# 5. SAVE RESULTS
# ===================================================================

# Save grouped data
top_quartile.to_csv('results/lape/ROME/en_top_performers.csv', index=False)
bottom_quartile.to_csv('results/lape/ROME/en_bottom_performers.csv', index=False)

print("\n✓ Results saved to CSV files")