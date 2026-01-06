#!/usr/bin/env python
"""
Generate Training Data for GMSH Curriculum Training

Creates parquet files with programmatic geometry data for curriculum agent training.
Each row contains geometry code, metadata, and formatted prompt for the curriculum agent.
"""

import argparse
import pandas as pd
import os
import sys

from programmatic_geometry import ProgrammaticGeometryGenerator, format_geometry_context


def generate_training_data(
    num_samples: int,
    output_path: str,
    seed: int = 42
):
    """
    Generate training data with programmatic geometries.

    Args:
        num_samples: Number of geometry samples to generate
        output_path: Path to save parquet file
        seed: Random seed for reproducibility
    """
    print(f"Generating {num_samples} geometries with seed {seed}...")

    generator = ProgrammaticGeometryGenerator(seed=seed)
    geometries = generator.generate_batch(num_samples)

    # Build data for parquet
    data = []
    for geom in geometries:
        # Format the geometry context as the prompt
        prompt = format_geometry_context(geom)

        data.append({
            # Prompt for curriculum agent
            "prompt": prompt,

            # Ground truth info (passed to reward function)
            "geometry_code": geom["geometry_code"],
            "geometry_name": geom["geometry_name"],
            "geometry_description": geom["geometry_description"],
            "expected_surfaces": geom["expected_surfaces"],
            "expected_volumes": geom["expected_volumes"],
            "characteristic_length": geom["characteristic_length"],
            "parameters": str(geom["parameters"]),  # Convert dict to string for parquet
        })

    # Create DataFrame and save
    df = pd.DataFrame(data)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")

    # Print sample
    print(f"\nSample data:")
    print(f"  Geometry: {data[0]['geometry_name']}")
    print(f"  Prompt:\n{data[0]['prompt'][:500]}...")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for GMSH curriculum training"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=5000,
        help="Number of training samples"
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=500,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Generate training data
    train_path = os.path.join(args.output_dir, "geometry_train.parquet")
    generate_training_data(
        num_samples=args.train_samples,
        output_path=train_path,
        seed=args.seed
    )

    # Generate validation data (different seed)
    val_path = os.path.join(args.output_dir, "geometry_val.parquet")
    generate_training_data(
        num_samples=args.val_samples,
        output_path=val_path,
        seed=args.seed + 1000  # Different seed for val
    )

    print(f"\nDone! Generated:")
    print(f"  Training: {train_path} ({args.train_samples} samples)")
    print(f"  Validation: {val_path} ({args.val_samples} samples)")


if __name__ == "__main__":
    main()
