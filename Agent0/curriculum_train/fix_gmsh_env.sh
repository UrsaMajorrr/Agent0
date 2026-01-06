#!/bin/bash
# Fix Gmsh JPEG/TIFF library conflict

echo "Fixing Gmsh library dependencies..."

# Option 1: Reinstall gmsh with conda-forge (usually has better compatibility)
conda install -c conda-forge gmsh -y

# Option 2: If that doesn't work, reinstall libjpeg-turbo
# conda install -c conda-forge libjpeg-turbo -y

# Option 3: Reinstall both gmsh and imaging libraries
# conda install -c conda-forge gmsh libjpeg-turbo libtiff -y

echo "Done! Try running the training script again."
