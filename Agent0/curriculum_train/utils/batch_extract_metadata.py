#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch process geometry files to extract metadata for all geometries in a directory.

Usage:
    python batch_extract_metadata.py --geometry_dir /path/to/geometries --output_file metadata_all.json
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from geometry_metadata_extract import extract_geometry_metadata


def batch_process_geometries(geometry_dir: str, output_file: str, pattern: str = "*.step"):
    """
    Process all geometry files in a directory and save combined metadata.

    Args:
        geometry_dir: Directory containing STEP files
        output_file: Output JSON file path
        pattern: File pattern to match (default: *.step)
    """
    geometry_dir = Path(geometry_dir)

    # Find all geometry files
    if pattern == "*.step":
        geometry_files = list(geometry_dir.glob("*.step")) + list(geometry_dir.glob("*.STEP"))
    else:
        geometry_files = list(geometry_dir.glob(pattern))

    if not geometry_files:
        print(f"No geometry files found in {geometry_dir}")
        return

    print(f"Found {len(geometry_files)} geometry files")

    all_metadata = []
    failed_files = []

    for geom_file in tqdm(geometry_files, desc="Extracting metadata"):
        try:
            metadata = extract_geometry_metadata(str(geom_file))
            all_metadata.append(metadata)
        except Exception as e:
            print(f"\nError processing {geom_file.name}: {e}")
            failed_files.append({"file": str(geom_file), "error": str(e)})

    # Save combined metadata
    output_data = {
        "total_geometries": len(geometry_files),
        "successful": len(all_metadata),
        "failed": len(failed_files),
        "failed_files": failed_files,
        "metadata": all_metadata
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Processed {len(all_metadata)}/{len(geometry_files)} geometries successfully")
    print(f"✓ Metadata saved to {output_file}")

    if failed_files:
        print(f"\n⚠ {len(failed_files)} files failed:")
        for failed in failed_files:
            print(f"  - {Path(failed['file']).name}: {failed['error']}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    if all_metadata:
        complexities = [m["complexity_score"] for m in all_metadata]
        multi_scales = [m["multi_scale_ratio"] for m in all_metadata]
        surface_counts = [m["surfaces"] for m in all_metadata]

        print(f"\nComplexity scores: min={min(complexities):.1f}, max={max(complexities):.1f}, avg={sum(complexities)/len(complexities):.1f}")
        print(f"Multi-scale ratios: min={min(multi_scales):.1f}, max={max(multi_scales):.1f}, avg={sum(multi_scales)/len(multi_scales):.1f}")
        print(f"Surface counts: min={min(surface_counts)}, max={max(surface_counts)}, avg={sum(surface_counts)/len(surface_counts):.1f}")

        # Show entity type distribution across all geometries
        all_surface_types = set()
        for m in all_metadata:
            all_surface_types.update(m["surface_type_distribution"].keys())
        print(f"\nUnique surface types across all geometries: {', '.join(sorted(all_surface_types))}")


def main():
    parser = argparse.ArgumentParser(description="Batch extract geometry metadata")
    parser.add_argument("--geometry_dir", type=str, required=True,
                       help="Directory containing geometry files")
    parser.add_argument("--output_file", type=str, default="metadata_all.json",
                       help="Output JSON file (default: metadata_all.json)")
    parser.add_argument("--pattern", type=str, default="*.step",
                       help="File pattern (default: *.step)")

    args = parser.parse_args()

    batch_process_geometries(args.geometry_dir, args.output_file, args.pattern)


if __name__ == "__main__":
    main()
