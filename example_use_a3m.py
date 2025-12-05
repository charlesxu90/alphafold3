#!/usr/bin/env python3
"""Simple example showing how to use A3M files with AlphaFold 3.

This is a minimal example demonstrating the basic workflow.
"""

import json
import os
from alphafold3.common import folding_input
import run_alphafold


def load_a3m(path):
    """Load an A3M file."""
    with open(path, 'r') as f:
        return f.read()


def main():
    # Step 1: Create base input JSON
    input_dict = {
        'name': 'my_protein',
        'modelSeeds': [1234],
        'sequences': [
            {
                'protein': {
                    'id': 'A',
                    'sequence': 'MKLLVLGLGSLLLLLLLA',
                    'modifications': [],
                }
            }
        ],
        'dialect': 'alphafold3',
        'version': 1,
    }
    
    # Step 2: Add A3M MSA data
    # Replace these paths with your actual A3M files
    if os.path.exists('unpaired.a3m'):
        input_dict['sequences'][0]['protein']['unpairedMsa'] = load_a3m('unpaired.a3m')
    
    if os.path.exists('paired.a3m'):
        input_dict['sequences'][0]['protein']['pairedMsa'] = load_a3m('paired.a3m')
    
    # Step 3: Create fold input
    input_json = json.dumps(input_dict)
    fold_input = folding_input.Input.from_json(input_json)
    
    # Step 4: Run AlphaFold 3 (without MSA search pipeline)
    # Note: You need to set up model_runner first
    # model_runner = run_alphafold.ModelRunner(...)
    
    # results = run_alphafold.process_fold_input(
    #     fold_input=fold_input,
    #     data_pipeline_config=None,  # Skip MSA search since we have MSAs
    #     model_runner=model_runner,
    #     output_dir='output/',
    # )
    
    print("Fold input created successfully!")
    print(f"Protein sequence: {fold_input.protein_chains[0].sequence}")
    print(f"Has unpaired MSA: {fold_input.protein_chains[0].unpaired_msa is not None}")
    print(f"Has paired MSA: {fold_input.protein_chains[0].paired_msa is not None}")


if __name__ == '__main__':
    main()
