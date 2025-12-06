#!/usr/bin/env python3
"""Prepare AlphaFold3 input configs for protein variants.

This script generates AlphaFold3 input JSON files for protein variants by:
1. Reading variant sequences from a FASTA file
2. Using a wild-type AF3 result JSON (containing processed MSA and templates)
3. Creating input.json files where the query sequence is replaced with the variant
   while preserving the MSA and templates from the wild-type

Usage:
  python prepare_variant_af3_configs.py \
    --variants_fasta=sequences.fasta \
    --wt_data_json=wt_data.json \
    --output_dir=output

  # With optional ligand
  python prepare_variant_af3_configs.py \
    --variants_fasta=sequences.fasta \
    --wt_data_json=wt_data.json \
    --output_dir=output \
    --ligand_smiles="CCO"

Example directory structure after running:
  output/
    seq_0/
      input.json
    seq_1/
      input.json
    ...
"""

import argparse
import copy
import json
import os
import re


def parse_fasta(fasta_path: str) -> list[tuple[str, str]]:
    """Parse a FASTA file and return list of (name, sequence) tuples."""
    sequences = []
    current_name = None
    current_seq_parts = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # Save previous sequence
                if current_name is not None:
                    sequences.append((current_name, ''.join(current_seq_parts)))
                # Start new sequence
                current_name = line[1:].split()[0]  # Get ID before first space
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

        # Don't forget the last sequence
        if current_name is not None:
            sequences.append((current_name, ''.join(current_seq_parts)))

    return sequences


def load_wt_data_json(json_path: str) -> dict:
    """Load and parse the wild-type AF3 data JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_protein_entity(wt_data: dict) -> tuple[dict, str]:
    """Extract the protein entity from the wild-type data.

    Returns:
        Tuple of (protein_entity_dict, wt_sequence)
    """
    for seq_entry in wt_data.get('sequences', []):
        if 'protein' in seq_entry:
            protein = seq_entry['protein']
            return protein, protein.get('sequence', '')
    raise ValueError("No protein entity found in wild-type data JSON")


def create_variant_msa(wt_msa: str, variant_name: str, variant_sequence: str) -> str:
    """Create a variant MSA by replacing the query sequence in the unpaired MSA.

    The MSA format is A3M-like with entries separated by '>' headers.
    The first entry is the query sequence which we replace with the variant.

    Args:
        wt_msa: The unpaired MSA string from wild-type data.
        variant_name: Name for the variant (used in header).
        variant_sequence: The variant protein sequence.

    Returns:
        MSA content as a string with the variant sequence as query.
    """
    # Split MSA into entries
    lines = wt_msa.split('\n')

    result_lines = []
    first_entry = True
    skip_sequence = False

    for line in lines:
        if line.startswith('>'):
            if first_entry:
                # Replace first header with variant name
                result_lines.append(f'>{variant_name}')
                skip_sequence = True
                first_entry = False
            else:
                result_lines.append(line)
                skip_sequence = False
        else:
            if skip_sequence:
                # Replace first sequence with variant
                result_lines.append(variant_sequence)
                skip_sequence = False
            else:
                result_lines.append(line)

    return '\n'.join(result_lines)


def create_variant_input_json(
    wt_data: dict,
    variant_name: str,
    variant_sequence: str,
    wt_sequence: str,
    ligand_smiles: str | None = None,
    ligand_id: str = 'B',
    model_seeds: list[int] | None = None,
) -> dict:
    """Create an AlphaFold3 input JSON for a variant based on wild-type data.

    This function creates a new input JSON by:
    1. Copying the wild-type data structure
    2. Replacing the protein sequence with the variant
    3. Modifying the MSA to use the variant as the query sequence
    4. Keeping the templates from wild-type (they remain valid for point mutations)
    5. Optionally adding a ligand

    Args:
        wt_data: The wild-type AF3 data JSON dictionary.
        variant_name: Name for the variant prediction job.
        variant_sequence: The variant protein sequence.
        wt_sequence: The wild-type protein sequence (for validation).
        ligand_smiles: Optional SMILES string for a ligand to add.
        ligand_id: Chain ID for the ligand.
        model_seeds: Random seeds for prediction (overrides wt_data if provided).

    Returns:
        Dictionary ready to be saved as JSON.
    """
    # Validate sequence lengths match
    if len(variant_sequence) != len(wt_sequence):
        raise ValueError(
            f"Variant sequence length ({len(variant_sequence)}) doesn't match "
            f"wild-type sequence length ({len(wt_sequence)}). "
            f"Only point mutations are supported."
        )

    # Deep copy to avoid modifying original
    result = copy.deepcopy(wt_data)

    # Update name
    result['name'] = variant_name

    # Update model seeds if provided
    if model_seeds is not None:
        result['modelSeeds'] = model_seeds

    # Find and update the protein entity
    protein_found = False
    for seq_entry in result.get('sequences', []):
        if 'protein' in seq_entry:
            protein = seq_entry['protein']
            protein_found = True

            # Update sequence
            protein['sequence'] = variant_sequence

            # Update unpaired MSA if present
            if 'unpairedMsa' in protein and protein['unpairedMsa']:
                protein['unpairedMsa'] = create_variant_msa(
                    protein['unpairedMsa'],
                    variant_name,
                    variant_sequence
                )

            # Templates are kept as-is since they're structural templates
            # and remain valid for point mutations

            break

    if not protein_found:
        raise ValueError("No protein entity found in wild-type data")

    # Add ligand if specified and not already present
    if ligand_smiles:
        # Check if ligand already exists
        has_ligand = any('ligand' in seq for seq in result.get('sequences', []))
        if not has_ligand:
            result['sequences'].append({
                'ligand': {
                    'id': ligand_id,
                    'smiles': ligand_smiles,
                }
            })

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Prepare AlphaFold3 input configs for protein variants.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--variants_fasta',
        required=True,
        help='Path to FASTA file containing variant sequences.',
    )
    parser.add_argument(
        '--wt_data_json',
        required=True,
        help='Path to wild-type AF3 data JSON file (contains processed MSA and templates).',
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for variant configs.',
    )
    parser.add_argument(
        '--ligand_smiles',
        default=None,
        help='Optional SMILES string for a ligand to include.',
    )
    parser.add_argument(
        '--ligand_id',
        default='B',
        help='Chain ID for the ligand (default: B).',
    )
    parser.add_argument(
        '--model_seeds',
        default=None,
        help='Comma-separated list of model seeds (default: use seeds from wt_data_json).',
    )

    args = parser.parse_args()

    # Parse model seeds if provided
    model_seeds = None
    if args.model_seeds:
        model_seeds = [int(s.strip()) for s in args.model_seeds.split(',')]

    # Load wild-type data JSON
    print(f'Loading wild-type data from: {args.wt_data_json}')
    wt_data = load_wt_data_json(args.wt_data_json)

    # Extract wild-type protein info
    wt_protein, wt_sequence = get_protein_entity(wt_data)
    print(f'Wild-type sequence: {wt_data.get("name", "unknown")} ({len(wt_sequence)} aa)')

    # Report templates if present
    templates = wt_protein.get('templates', [])
    if templates:
        print(f'Templates: {len(templates)} structure(s)')

    # Report MSA if present
    unpaired_msa = wt_protein.get('unpairedMsa', '')
    if unpaired_msa:
        msa_count = unpaired_msa.count('>')
        print(f'Unpaired MSA: {msa_count} sequences')

    # Load variant sequences
    variant_sequences = parse_fasta(args.variants_fasta)
    if not variant_sequences:
        raise ValueError(f'No sequences found in variants FASTA: {args.variants_fasta}')
    print(f'Loaded {len(variant_sequences)} variant sequences')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each variant
    created_count = 0
    skipped_count = 0

    for variant_name, variant_sequence in variant_sequences:
        # Check sequence length matches
        if len(variant_sequence) != len(wt_sequence):
            print(f'  Warning: Skipping {variant_name} - length mismatch '
                  f'({len(variant_sequence)} vs {len(wt_sequence)} aa)')
            skipped_count += 1
            continue

        # Create variant directory
        variant_dir = os.path.join(args.output_dir, variant_name)
        os.makedirs(variant_dir, exist_ok=True)

        # Create input JSON with MSA and templates
        input_json = create_variant_input_json(
            wt_data=wt_data,
            variant_name=variant_name,
            variant_sequence=variant_sequence,
            wt_sequence=wt_sequence,
            ligand_smiles=args.ligand_smiles,
            ligand_id=args.ligand_id,
            model_seeds=model_seeds,
        )

        # Save input JSON
        json_path = os.path.join(variant_dir, 'input.json')
        with open(json_path, 'w') as f:
            json.dump(input_json, f, indent=2)

        created_count += 1

    print(f'\nCreated {created_count} variant configs in {args.output_dir}')
    if skipped_count > 0:
        print(f'Skipped {skipped_count} variants due to length mismatch')

    # Print example usage
    print('\nTo run AlphaFold3 on a variant:')
    print(f'  python run_alphafold.py \\')
    print(f'    --json_path={args.output_dir}/<variant_name>/input.json \\')
    print(f'    --output_dir={args.output_dir}/<variant_name> \\')
    print(f'    --model_dir=<path_to_model>')


if __name__ == '__main__':
    main()
