import pandas as pd
import json
import random

# Load metadata
with open('metadata_all.json') as f:
    metadata = json.load(f)

# Create dataset - replicate each geometry 50 times
# Curriculum agent will generate different tasks each time
TASKS_PER_GEOMETRY = 50

data = []
for meta in metadata['metadata']:
    # Add this geometry 50 times
    for _ in range(TASKS_PER_GEOMETRY):
        # Format surface types as compact string
        surf_types = meta.get('surface_type_distribution', {})
        surface_types_str = ', '.join([f"{count} {stype}" for stype, count in surf_types.items()])

        data.append({
            'geometry_file': meta['file'],
            # Minimal metadata for curriculum agent
            'surfaces': meta.get('surfaces', 0),
            'curves': meta.get('curves', 0),
            'volumes': meta.get('volumes', 0),
            'surface_types': surface_types_str,  # e.g., "20 Plane, 6 Cylinder"
            'entity_types': meta.get('entity_types', {}),
            'overall_bbox': meta.get('overall_bbox', {}),
            'multi_scale_ratio': round(meta.get('multi_scale_ratio', 1.0), 2),
            'characteristic_length': round(meta.get('characteristic_length', 1.0), 2),
        })

print(f"Total tasks: {len(data)} ({len(metadata['metadata'])} geometries × {TASKS_PER_GEOMETRY})")

# Shuffle
random.seed(42)
random.shuffle(data)

# Split 90/10
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
val_data = data[split_idx:]

# Save as parquet
pd.DataFrame(train_data).to_parquet('geometry_train.parquet', index=False)
pd.DataFrame(val_data).to_parquet('geometry_val.parquet', index=False)

print(f"Train: {len(train_data)} tasks")
print(f"Val: {len(val_data)} tasks")
