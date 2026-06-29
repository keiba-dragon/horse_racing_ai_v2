import pickle, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('models/conditional_logit.pkl', 'rb') as f:
    m = pickle.load(f)

print("keys:", list(m.keys()))
coef = np.array(m['coef'])
print(f"coef shape: {coef.shape}")
print(f"feat_cols: {len(m['feat_cols'])}")
print(f"top_idx is None: {m['top_idx'] is None}")
if m['top_idx'] is not None:
    print(f"top_idx len: {len(m['top_idx'])}")
