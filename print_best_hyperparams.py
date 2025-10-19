import csv, json, os
csv_path = os.path.join('results', 'hyper_search_20251019_001851.csv')
if not os.path.exists(csv_path):
    raise SystemExit(f'CSV not found: {csv_path}')

best = {}
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        model = row['model']
        mf = float(row['micro_f1']) if row.get('micro_f1') else -1.0
        if model not in best or mf > best[model][0]:
            best[model] = (mf, row)

out = {}
for m, (mf, cfg) in best.items():
    # extract relevant fields
    config = {k: cfg[k] for k in cfg.keys() if k not in ('micro_f1','macro_f1','accuracy')}
    out[m] = {'micro_f1': mf, 'config': config}

print(json.dumps(out, indent=2))
