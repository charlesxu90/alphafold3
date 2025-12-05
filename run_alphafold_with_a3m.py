#!/usr/bin/env python3
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Example script for running AlphaFold 3 with pre-computed A3M MSA files.

This script demonstrates how to use pre-computed A3M files instead of running
the full MSA search pipeline. This is useful when:
1. You already have MSAs from external sources (e.g., MMseqs2, HHblits)
2. You want to reuse previously computed MSAs
3. You want to test different MSAs without re-running the search

Usage:
  python run_alphafold_with_a3m.py \
    --input_json=path/to/input.json \
    --unpaired_msa_a3m=path/to/unpaired.a3m \
    --paired_msa_a3m=path/to/paired.a3m \
    --output_dir=path/to/output \
    --model_dir=path/to/model

A3M file format:
  A3M files are FASTA-like files with MSA information. Example:
  
  >query_sequence
  MKLLVLGLGSLLLLLLLLLL
  >hit1
  MKLLVLGLGS--LLLLLLLL
  >hit2
  MKLLVLGLGSllLLLLLLLL
  
  Lowercase letters indicate insertions relative to the query.
  Dashes (-) indicate deletions.
"""

import json
import os
import pathlib
from typing import Any

from absl import app
from absl import flags
from absl import logging
from alphafold3.common import folding_input
import jax

import run_alphafold


FLAGS = flags.FLAGS

flags.DEFINE_string('input_json', None, 'Path to input JSON file')
flags.DEFINE_string(
    'unpaired_msa_a3m',
    None,
    'Path to unpaired MSA A3M file (optional, can specify per-chain)',
)
flags.DEFINE_string(
    'paired_msa_a3m',
    None,
    'Path to paired MSA A3M file (optional, can specify per-chain)',
)
flags.DEFINE_string(
    'unpaired_msa_dir',
    None,
    'Directory containing unpaired A3M files named {chain_id}.a3m (optional)',
)
flags.DEFINE_string(
    'paired_msa_dir',
    None,
    'Directory containing paired A3M files named {chain_id}.a3m (optional)',
)
# flags.DEFINE_string('output_dir', None, 'Path to output directory')
# flags.DEFINE_string(
#     'model_dir',
#     run_alphafold.DEFAULT_MODEL_DIR,
#     'Path to model parameters',
# )
# flags.DEFINE_list(
#     'buckets',
#     None,
#     'Comma-separated list of token bucket sizes (e.g., "256,512,1024")',
# )
# flags.DEFINE_bool(
#     'run_inference',
#     True,
#     'Whether to run inference (requires model parameters)',
# )


def load_a3m_file(a3m_path: str) -> str:
  """Load A3M file and return its contents as a string."""
  with open(a3m_path, 'r') as f:
    return f.read()


def add_a3m_to_fold_input(
    fold_input_json: str,
    unpaired_msa_path: str | None = None,
) -> str:
  """Add A3M MSA data to a fold input JSON.
  
  Args:
    fold_input_json: JSON string containing the fold input.
    unpaired_msa_path: Path to a single unpaired A3M file (applied to first protein).
    
  Returns:
    Modified JSON string with MSA data included.
  """
  fold_input_dict = json.loads(fold_input_json)
  
  # Track if we've applied single-file MSAs
  applied_single_unpaired = False
  applied_single_paired = False
  
  # Process each sequence
  for seq in fold_input_dict['sequences']:
    # Handle protein sequences
    if 'protein' in seq:
      protein = seq['protein']
      raw_chain_id = protein['id']
      # Handle both string and list formats for chain_id
      chain_id = raw_chain_id[0] if isinstance(raw_chain_id, list) else raw_chain_id
      
      # Try to load unpaired MSA
      unpaired_msa_content = None
      if unpaired_msa_dir and chain_id:
        chain_a3m_path = os.path.join(unpaired_msa_dir, f'{chain_id}.a3m')
        if os.path.exists(chain_a3m_path):
          unpaired_msa_content = load_a3m_file(chain_a3m_path)
          logging.info(f'Loaded unpaired MSA for chain {chain_id} from {chain_a3m_path}')
      elif unpaired_msa_path and not applied_single_unpaired:
        unpaired_msa_content = load_a3m_file(unpaired_msa_path)
        applied_single_unpaired = True
        logging.info(f'Loaded unpaired MSA from {unpaired_msa_path}')
      
      if unpaired_msa_content:
        protein['unpairedMsa'] = unpaired_msa_content
      
  
  return json.dumps(fold_input_dict)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  # Validate required flags
  if not FLAGS.input_json:
    raise app.UsageError('--input_json is required')
  if not FLAGS.output_dir:
    raise app.UsageError('--output_dir is required')
  
  # Check if at least one MSA source is provided
  has_msa_source = (
      FLAGS.unpaired_msa_a3m
  )
  
  if not has_msa_source:
    logging.warning(
        'No A3M files specified. The input JSON must already contain MSAs, '
        'or MSA search pipeline will need to be run.'
    )
  
  # Load input JSON
  logging.info(f'Loading input from {FLAGS.input_json}')
  with open(FLAGS.input_json, 'r') as f:
    input_json = f.read()
  
  # Add A3M data if provided
  if has_msa_source:
    logging.info('Adding A3M MSA data to input')
    input_json = add_a3m_to_fold_input(
        input_json,
        unpaired_msa_path=FLAGS.unpaired_msa_a3m,
    )
  
  # Parse fold input
  fold_input = folding_input.Input.from_json(input_json)
  logging.info(f'Processing fold input: {fold_input.name}')
  
  # Create output directory
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  
  # Setup model runner if inference is requested
  model_runner = None
  if FLAGS.run_inference:
    logging.info('Setting up model runner')
    model_runner = run_alphafold.ModelRunner(
        model_class=run_alphafold.diffusion_model.Diffuser,
        config=run_alphafold.make_model_config(),
        device=jax.local_devices()[0],
        model_dir=pathlib.Path(FLAGS.model_dir),
    )
  
  # Parse buckets if provided
  buckets = None
  if FLAGS.buckets:
    buckets = [int(b) for b in FLAGS.buckets]
    logging.info(f'Using token buckets: {buckets}')
  
  # Run AlphaFold 3
  # Note: data_pipeline_config is None since we're providing MSAs directly
  logging.info('Running AlphaFold 3')
  results = run_alphafold.process_fold_input(
      fold_input=fold_input,
      data_pipeline_config=None,  # Skip MSA search pipeline
      model_runner=model_runner,
      output_dir=FLAGS.output_dir,
      buckets=buckets,
  )
  
  if results and isinstance(results, list):
    logging.info(f'Successfully generated {len(results)} predictions')
    logging.info(f'Results saved to {FLAGS.output_dir}')
  else:
    logging.info('No inference results (data pipeline only)')
  
  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
