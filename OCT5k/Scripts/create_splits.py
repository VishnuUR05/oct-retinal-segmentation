import os
import glob
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def get_stats(df, name):
    groups = df['e2e_group_id'].nunique()
    imgs = len(df)
    cats = df['category'].value_counts().to_dict()
    return groups, imgs, cats

def create_splits():
    dataset_root = os.path.abspath('.')
    images_root = os.path.join(dataset_root, 'Images', 'Images_Manual')
    masks_root = os.path.join(dataset_root, 'Masks', 'Masks_Manual', 'Grading_1')
    bounds_root = os.path.join(dataset_root, 'Boundaries', 'Boundaries_Manual', 'Grading_1')
    
    all_images = glob.glob(os.path.join(images_root, '**', '*.png'), recursive=True)
    
    # Extract metadata for each image
    data = []
    e2e_to_category = {}
    
    for img_path in all_images:
        parts = img_path.replace('\\', '/').split('/')
        category = parts[-4]
        e2e = parts[-3]
        date = parts[-2]
        img_file = parts[-1]
        
        slice_id = img_file.replace('.png', '')
        sample_id = f"{e2e}_{slice_id}"
        
        mask_path = os.path.join(masks_root, category, e2e, date, img_file)
        bound_path = os.path.join(bounds_root, category, e2e, date, img_file.replace('.png', '.csv'))
        
        data.append({
            'sample_id': sample_id,
            'image_path': img_path,
            'mask_path': mask_path,
            'boundary_path': bound_path,
            'category': category,
            'e2e_group_id': e2e,
            'slice_id': slice_id
        })
        e2e_to_category[e2e] = category
        
    df = pd.DataFrame(data)
    
    # Stratified split by group
    # We have e2e_to_category, let's group by category and split the e2es
    categories = set(e2e_to_category.values())
    
    train_e2es, val_e2es, test_e2es = [], [], []
    
    random.seed(42)
    np.random.seed(42)
    
    for cat in sorted(list(categories)):
        cat_e2es = sorted([e for e, c in e2e_to_category.items() if c == cat])
        random.shuffle(cat_e2es)
        
        n = len(cat_e2es)
        n_train = int(np.round(0.7 * n))
        n_val = int(np.round(0.15 * n))
        # If sizes are too small, ensure at least some representation if possible
        if n_val == 0 and n > 1:
            n_val = 1
        n_test = n - n_train - n_val
        
        train_e2es.extend(cat_e2es[:n_train])
        val_e2es.extend(cat_e2es[n_train:n_train+n_val])
        test_e2es.extend(cat_e2es[n_train+n_val:])
        
    train_df = df[df['e2e_group_id'].isin(train_e2es)]
    val_df = df[df['e2e_group_id'].isin(val_e2es)]
    test_df = df[df['e2e_group_id'].isin(test_e2es)]
    
    # Save CSVs
    splits_dir = os.path.join(dataset_root, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)
    
    # Generate Stats
    tr_gr, tr_im, tr_c = get_stats(train_df, 'Train')
    va_gr, va_im, va_c = get_stats(val_df, 'Val')
    te_gr, te_im, te_c = get_stats(test_df, 'Test')
    
    tot_gr = tr_gr + va_gr + te_gr
    tot_im = tr_im + va_im + te_im
    
    report = []
    report.append("OCT5k Dataset Split Report")
    report.append("==========================")
    report.append(f"Total E2E Groups: {tot_gr}")
    report.append(f"Total Images: {tot_im}\n")
    
    for name, gr, im, c in [('Train', tr_gr, tr_im, tr_c), ('Validation', va_gr, va_im, va_c), ('Test', te_gr, te_im, te_c)]:
        report.append(f"--- {name} Split ---")
        report.append(f"E2E Groups: {gr} ({gr/tot_gr*100:.1f}%)")
        report.append(f"Images: {im} ({im/tot_im*100:.1f}%)")
        report.append(f"Images per category:")
        for cat, count in c.items():
            report.append(f"  - {cat}: {count}")
        report.append("")
        
    out_dir = os.path.join(dataset_root, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'split_report.txt'), 'w') as f:
        f.write("\n".join(report))
        
    # Plotting
    labels = list(categories)
    tr_counts = [tr_c.get(l, 0) for l in labels]
    va_counts = [va_c.get(l, 0) for l in labels]
    te_counts = [te_c.get(l, 0) for l in labels]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10,6))
    rects1 = ax.bar(x - width, tr_counts, width, label='Train')
    rects2 = ax.bar(x, va_counts, width, label='Validation')
    rects3 = ax.bar(x + width, te_counts, width, label='Test')
    
    ax.set_ylabel('Number of Images')
    ax.set_title('Category Distribution Across Splits')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, 'split_distribution.png'))
    plt.close()

if __name__ == "__main__":
    create_splits()
    print("Splits created successfully!")
