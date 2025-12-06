#!/usr/bin/env python3
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for running AlphaFold 3 with pre-computed A3M MSA files.

This script allows using pre-computed A3M files instead of running the full
MSA search pipeline. Template search can optionally be enabled to find
structural templates using the provided MSA.

Use cases:
1. You already have MSAs from external sources (e.g., MMseqs2, HHblits)
2. You want to reuse previously computed MSAs
3. You want to test different MSAs without re-running the search
4. You want to skip MSA search but still run template search

Usage:
  # Option 1: With MSA and template search (recommended for best results)
  python run_alphafold_with_a3m.py \
    --input_json=path/to/input.json \
    --output_dir=path/to/output \
    --model_dir=path/to/model \
    --run_template_search=true \
    --db_dir=/path/to/databases

  # Option 2: With MSA only (no template search, faster)
  python run_alphafold_with_a3m.py \
    --input_json=path/to/input.json \
    --output_dir=path/to/output \
    --model_dir=path/to/model \
    --run_template_search=false

  Input JSON format with msa_path:
  {
    "sequences": [
      {
        "protein": {
          "id": ["A"],
          "sequence": "MKLLVLGLGS...",
          "msa_path": "path/to/protein.a3m"  # Relative to input.json or absolute
        }
      }
    ]
  }

  # Option 3: Specify A3M files via command-line flags
  python run_alphafold_with_a3m.py \
    --input_json=path/to/input.json \
    --unpaired_msa_a3m=path/to/unpaired.a3m \
    --output_dir=path/to/output \
    --model_dir=path/to/model

Template Search:
  By default, template search is enabled (--run_template_search=true).
  This requires the following databases to be configured:
  - seqres_database_path: PDB sequence database (pdb_seqres_*.fasta)
  - pdb_database_path: PDB structure database (mmCIF files)

  And the following binaries:
  - hmmsearch_binary_path
  - hmmbuild_binary_path

  Set --run_template_search=false to skip template search entirely.

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

import datetime
import json
import os
import pathlib
import string
from typing import Any

from absl import app
from absl import flags
from absl import logging
from alphafold3.common import folding_input
from alphafold3.constants import mmcif_names
from alphafold3.data import msa_config
from alphafold3.data import structure_stores
from alphafold3.data import templates
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
flags.DEFINE_bool(
    'run_template_search',
    True,
    'Whether to run template search using the provided MSA. If False, no '
    'templates will be used. Requires database paths to be configured.',
)
flags.DEFINE_string(
    'max_template_date',
    '2021-09-30',
    'Maximum template release date (YYYY-MM-DD format). Templates released '
    'after this date will be excluded.',
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


def clean_a3m_content(a3m_content: str) -> str:
  """Clean A3M content to be compatible with AlphaFold3.

  This function:
  1. Removes metadata from header lines (keeps only sequence ID)
  2. Replaces invalid amino acid characters with gaps (-)
  3. Removes sequences with too many invalid characters
  4. Filters sequences that don't match query alignment length

  In A3M format:
  - Uppercase letters and gaps (-) are aligned positions
  - Lowercase letters are insertions (don't count toward alignment length)
  - All sequences must have the same number of aligned positions as the query

  Args:
    a3m_content: Raw A3M file content.

  Returns:
    Cleaned A3M content compatible with AlphaFold3.
  """
  # Valid amino acid characters (uppercase for aligned, lowercase for insertions)
  valid_upper = set('ACDEFGHIKLMNPQRSTVWY-')
  valid_lower = set('acdefghiklmnpqrstvwy')
  valid_aa = valid_upper | valid_lower

  lines = a3m_content.strip().split('\n')
  sequences = []  # List of (header, sequence) tuples
  current_header = None
  current_seq_parts = []

  def count_aligned_positions(seq: str) -> int:
    """Count non-lowercase characters (uppercase + gaps = aligned positions)."""
    return sum(1 for c in seq if c.isupper() or c == '-')

  def process_sequence(header: str, seq_parts: list[str]) -> tuple[str, str] | None:
    """Process a header and sequence, return cleaned version or None if invalid."""
    if not header or not seq_parts:
      return None

    # Clean header: keep only the ID part (before first whitespace/tab)
    clean_header = header.split()[0] if header else header

    # Join and clean sequence
    full_seq = ''.join(seq_parts)

    # Replace invalid characters with gaps (for uppercase) or remove (for lowercase positions)
    cleaned_seq = []
    invalid_count = 0
    for char in full_seq:
      if char in valid_aa:
        cleaned_seq.append(char)
      elif char.isupper() or char == '-':
        # Invalid uppercase character - replace with gap
        cleaned_seq.append('-')
        invalid_count += 1
      else:
        # Invalid lowercase character - skip it (it's an insertion anyway)
        invalid_count += 1

    # Skip sequences with too many invalid characters (>10% of sequence)
    if len(full_seq) > 0 and invalid_count / len(full_seq) > 0.1:
      logging.debug(f'Skipping sequence {clean_header} with {invalid_count}/{len(full_seq)} invalid characters')
      return None

    return clean_header, ''.join(cleaned_seq)

  # First pass: collect all sequences
  for line in lines:
    if line.startswith('>'):
      # Process previous sequence if exists
      if current_header is not None:
        result = process_sequence(current_header, current_seq_parts)
        if result:
          sequences.append(result)

      # Start new sequence
      current_header = line
      current_seq_parts = []
    else:
      # Accumulate sequence lines
      current_seq_parts.append(line)

  # Don't forget the last sequence
  if current_header is not None:
    result = process_sequence(current_header, current_seq_parts)
    if result:
      sequences.append(result)

  if not sequences:
    return ''

  # Second pass: filter sequences by alignment length
  # The first sequence is the query - all others must match its aligned length
  query_header, query_seq = sequences[0]
  query_aligned_len = count_aligned_positions(query_seq)

  cleaned_lines = [query_header, query_seq]
  filtered_count = 0

  for header, seq in sequences[1:]:
    aligned_len = count_aligned_positions(seq)
    if aligned_len == query_aligned_len:
      cleaned_lines.append(header)
      cleaned_lines.append(seq)
    else:
      filtered_count += 1
      logging.debug(f'Filtered sequence {header}: aligned length {aligned_len} != query length {query_aligned_len}')

  if filtered_count > 0:
    logging.info(f'Filtered {filtered_count} sequences with mismatched alignment length (kept {len(sequences) - filtered_count} of {len(sequences)})')

  return '\n'.join(cleaned_lines)


def search_templates_with_msa(
    sequence: str,
    msa_a3m: str,
    max_template_date: datetime.date,
    seqres_database_path: str,
    pdb_database_path: str,
    hmmsearch_binary_path: str,
    hmmbuild_binary_path: str,
) -> list[dict]:
  """Search for templates using a provided MSA.

  Args:
    sequence: The protein sequence.
    msa_a3m: The MSA in A3M format to use for template search.
    max_template_date: Maximum template release date.
    seqres_database_path: Path to PDB sequence database.
    pdb_database_path: Path to PDB structure database.
    hmmsearch_binary_path: Path to hmmsearch binary.
    hmmbuild_binary_path: Path to hmmbuild binary.

  Returns:
    List of template dictionaries in folding_input format.
  """
  logging.info('Searching for templates using provided MSA...')

  # Configure hmmsearch
  hmmsearch_config = msa_config.HmmsearchConfig(
      hmmsearch_binary_path=hmmsearch_binary_path,
      hmmbuild_binary_path=hmmbuild_binary_path,
      filter_f1=0.1,
      filter_f2=0.1,
      filter_f3=0.1,
      e_value=100,
      inc_e=100,
      dom_e=100,
      incdom_e=100,
      alphabet='amino',
  )

  # Create structure store
  structure_store = structure_stores.StructureStore(pdb_database_path)

  # Search for templates
  template_hits = templates.Templates.from_seq_and_a3m(
      query_sequence=sequence,
      msa_a3m=msa_a3m,
      max_template_date=max_template_date,
      database_path=seqres_database_path,
      hmmsearch_config=hmmsearch_config,
      max_a3m_query_sequences=None,
      chain_poly_type=mmcif_names.PROTEIN_CHAIN,
      structure_store=structure_store,
  )

  # Filter templates
  filter_config = msa_config.TemplateFilterConfig(
      max_subsequence_ratio=0.95,
      min_align_ratio=0.1,
      min_hit_length=10,
      deduplicate_sequences=True,
      max_hits=4,
      max_template_date=max_template_date,
  )

  filtered_templates = template_hits.filter(
      max_subsequence_ratio=filter_config.max_subsequence_ratio,
      min_align_ratio=filter_config.min_align_ratio,
      min_hit_length=filter_config.min_hit_length,
      deduplicate_sequences=filter_config.deduplicate_sequences,
      max_hits=filter_config.max_hits,
  )

  # Convert to folding_input format
  # AlphaFold3 expects templates with 'queryIndices' and 'templateIndices' arrays
  template_list = []
  for hit, struc in filtered_templates.get_hits_with_structures():
    # Convert mapping dict to separate query/template index lists
    query_indices = []
    template_indices = []
    for query_idx, template_idx in hit.query_to_hit_mapping.items():
      query_indices.append(query_idx)
      template_indices.append(template_idx)

    template_list.append({
        'mmcif': struc.to_mmcif(),
        'queryIndices': query_indices,
        'templateIndices': template_indices,
    })

  logging.info(f'Found {len(template_list)} templates')
  return template_list


def add_a3m_to_fold_input(
    fold_input_json: str,
    unpaired_msa_path: str | None = None,
    unpaired_msa_dir: str | None = None,
    input_json_dir: str | None = None,
    run_template_search: bool = False,
    max_template_date: datetime.date | None = None,
    seqres_database_path: str | None = None,
    pdb_database_path: str | None = None,
    hmmsearch_binary_path: str | None = None,
    hmmbuild_binary_path: str | None = None,
) -> str:
  """Add A3M MSA data to a fold input JSON.

  Args:
    fold_input_json: JSON string containing the fold input.
    unpaired_msa_path: Path to a single unpaired A3M file (applied to first protein).
    unpaired_msa_dir: Directory containing unpaired A3M files named {chain_id}.a3m.
    input_json_dir: Directory of the input JSON file (for resolving relative msa_path).
    run_template_search: Whether to search for templates using the provided MSA.
    max_template_date: Maximum template release date (required if run_template_search).
    seqres_database_path: Path to PDB sequence database (required if run_template_search).
    pdb_database_path: Path to PDB structure database (required if run_template_search).
    hmmsearch_binary_path: Path to hmmsearch binary (required if run_template_search).
    hmmbuild_binary_path: Path to hmmbuild binary (required if run_template_search).

  Returns:
    Modified JSON string with MSA data included.
  """
  fold_input_dict = json.loads(fold_input_json)

  # Track if we've applied single-file MSAs
  applied_single_unpaired = False

  # Process each sequence
  for seq in fold_input_dict['sequences']:
    # Handle protein sequences
    if 'protein' in seq:
      protein = seq['protein']
      raw_chain_id = protein['id']
      # Handle both string and list formats for chain_id
      chain_id = raw_chain_id[0] if isinstance(raw_chain_id, list) else raw_chain_id

      # Try to load unpaired MSA from various sources (in order of priority):
      # 1. msa_path field in the protein definition (from input JSON)
      # 2. Chain-specific A3M file from unpaired_msa_dir
      # 3. Single unpaired_msa_path (applied to first protein only)
      unpaired_msa_content = None

      # Priority 1: Check for msa_path in protein definition
      if 'msa_path' in protein:
        msa_path = protein['msa_path']
        # Handle relative paths - resolve relative to input JSON directory
        if not os.path.isabs(msa_path) and input_json_dir:
          msa_path = os.path.join(input_json_dir, msa_path)
        if os.path.exists(msa_path):
          raw_content = load_a3m_file(msa_path)
          unpaired_msa_content = clean_a3m_content(raw_content)
          num_sequences = unpaired_msa_content.count('>')
          logging.info(f'Loaded and cleaned MSA for chain {chain_id} from msa_path: {msa_path} ({num_sequences} sequences)')
          # Remove msa_path from protein dict as it's not part of the standard format
          del protein['msa_path']
        else:
          logging.warning(f'MSA file not found: {msa_path}')

      # Priority 2: Chain-specific A3M from directory
      elif unpaired_msa_dir and chain_id:
        chain_a3m_path = os.path.join(unpaired_msa_dir, f'{chain_id}.a3m')
        if os.path.exists(chain_a3m_path):
          raw_content = load_a3m_file(chain_a3m_path)
          unpaired_msa_content = clean_a3m_content(raw_content)
          logging.info(f'Loaded and cleaned unpaired MSA for chain {chain_id} from {chain_a3m_path}')

      # Priority 3: Single unpaired MSA file (first protein only)
      elif unpaired_msa_path and not applied_single_unpaired:
        raw_content = load_a3m_file(unpaired_msa_path)
        unpaired_msa_content = clean_a3m_content(raw_content)
        applied_single_unpaired = True
        logging.info(f'Loaded and cleaned unpaired MSA from {unpaired_msa_path}')

      if unpaired_msa_content:
        protein['unpairedMsa'] = unpaired_msa_content
        # If pairedMsa is not set, set it to empty string (no paired MSA)
        if 'pairedMsa' not in protein:
          protein['pairedMsa'] = ''

        # Handle templates: either search or set to empty
        if 'templates' not in protein:
          if run_template_search and all([
              max_template_date,
              seqres_database_path,
              pdb_database_path,
              hmmsearch_binary_path,
              hmmbuild_binary_path,
          ]):
            # Run template search using the provided MSA
            try:
              template_list = search_templates_with_msa(
                  sequence=protein['sequence'],
                  msa_a3m=unpaired_msa_content,
                  max_template_date=max_template_date,
                  seqres_database_path=seqres_database_path,
                  pdb_database_path=pdb_database_path,
                  hmmsearch_binary_path=hmmsearch_binary_path,
                  hmmbuild_binary_path=hmmbuild_binary_path,
              )
              protein['templates'] = template_list
            except Exception as e:
              logging.warning(f'Template search failed for chain {chain_id}: {e}')
              logging.warning('Using empty templates list instead')
              protein['templates'] = []
          else:
            # No template search - use empty list
            protein['templates'] = []


  return json.dumps(fold_input_dict)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  # Validate required flags
  if not FLAGS.input_json:
    raise app.UsageError('--input_json is required')
  if not FLAGS.output_dir:
    raise app.UsageError('--output_dir is required')
  
  # Get the directory of the input JSON for resolving relative paths
  input_json_dir = os.path.dirname(os.path.abspath(FLAGS.input_json))

  # Check if at least one MSA source is provided (including msa_path in JSON)
  has_msa_source = (
      FLAGS.unpaired_msa_a3m or
      FLAGS.unpaired_msa_dir
  )

  # Also check if msa_path is specified in the input JSON
  with open(FLAGS.input_json, 'r') as f:
    input_check = json.load(f)
  for seq in input_check.get('sequences', []):
    if 'protein' in seq and 'msa_path' in seq['protein']:
      has_msa_source = True
      break

  if not has_msa_source:
    logging.warning(
        'No A3M files specified. The input JSON must already contain MSAs, '
        'or MSA search pipeline will need to be run.'
    )

  # Parse max_template_date
  max_template_date = None
  if FLAGS.run_template_search and FLAGS.max_template_date:
    try:
      max_template_date = datetime.datetime.strptime(
          FLAGS.max_template_date, '%Y-%m-%d'
      ).date()
    except ValueError:
      raise app.UsageError(
          f'Invalid max_template_date format: {FLAGS.max_template_date}. '
          'Expected YYYY-MM-DD.'
      )

  # Helper to expand ${DB_DIR} in database paths
  def expand_db_path(path: str) -> str:
    if path and '${DB_DIR}' in path:
      return string.Template(path).substitute(DB_DIR=FLAGS.db_dir)
    return path

  # Expand database paths
  seqres_database_path = expand_db_path(FLAGS.seqres_database_path)
  pdb_database_path = expand_db_path(FLAGS.pdb_database_path)

  # Check template search requirements
  if FLAGS.run_template_search:
    if not seqres_database_path or not pdb_database_path:
      logging.warning(
          'Template search requested but database paths not configured. '
          'Skipping template search. Set --seqres_database_path and '
          '--pdb_database_path to enable template search.'
      )
    if not FLAGS.hmmsearch_binary_path or not FLAGS.hmmbuild_binary_path:
      logging.warning(
          'Template search requested but hmmsearch/hmmbuild binaries not found. '
          'Skipping template search.'
      )

  # Load input JSON
  logging.info(f'Loading input from {FLAGS.input_json}')
  with open(FLAGS.input_json, 'r') as f:
    input_json = f.read()

  # Add A3M data if provided
  if has_msa_source:
    logging.info('Adding A3M MSA data to input')
    if FLAGS.run_template_search:
      logging.info('Template search enabled')
    else:
      logging.info('Template search disabled - using empty templates')
    input_json = add_a3m_to_fold_input(
        input_json,
        unpaired_msa_path=FLAGS.unpaired_msa_a3m,
        unpaired_msa_dir=FLAGS.unpaired_msa_dir,
        input_json_dir=input_json_dir,
        run_template_search=FLAGS.run_template_search,
        max_template_date=max_template_date,
        seqres_database_path=seqres_database_path,
        pdb_database_path=pdb_database_path,
        hmmsearch_binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
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
