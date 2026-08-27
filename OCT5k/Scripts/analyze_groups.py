import os
import glob
from collections import defaultdict
import numpy as np

root = 'Images/Images_Manual'
files = glob.glob(os.path.join(root, '**', '*.png'), recursive=True)

# Grouping by E2E identifier
e2e_groups = defaultdict(list)
e2e_to_category = defaultdict(set)
patient_to_e2e = defaultdict(set)

for f in files:
    # Path format: Images_Manual / Category / E2E / Date / Image.png
    parts = f.replace('\\', '/').split('/')
    if len(parts) >= 6:
        category = parts[-4]
        e2e = parts[-3]
        date = parts[-2]
        img = parts[-1]
        
        group_key = f'{category}/{e2e}'
        e2e_groups[group_key].append(f)
        e2e_to_category[e2e].add(category)

unique_e2e = set(k.split('/')[1] for k in e2e_groups.keys())
print(f'Total images: {len(files)}')
print(f'Unique Categories + E2E groups: {len(e2e_groups)}')
print(f'Unique E2E names (ignoring category): {len(unique_e2e)}')

# Check for E2E crossing categories
cross_cat = {e2e: cats for e2e, cats in e2e_to_category.items() if len(cats) > 1}
print(f'E2E groups crossing categories: {len(cross_cat)}')
if len(cross_cat) > 0:
    for k,v in cross_cat.items():
        print(f"  - {k}: {v}")

# Stats per group
counts = [len(imgs) for imgs in e2e_groups.values()]
print(f'Images per group: min={np.min(counts)}, max={np.max(counts)}, mean={np.mean(counts):.2f}, median={np.median(counts)}')
