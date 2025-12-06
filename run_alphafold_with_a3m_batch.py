#!/usr/bin/env python3
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Batch script for running AlphaFold 3 on multiple variants with pre-computed MSA/templates.

This script processes multiple input.json files (e.g., from prepare_variant_af3_configs.py)
in batch mode. The model is loaded once and reused for all predictions.

The script automatically detects the input format:
1. If input JSON already contains unpairedMsa and templates (from wt_data_json),
   it skips MSA/template search and runs inference directly.
2. If input JSON contains msa_path references, it loads the A3M files and
   optionally searches for templates.

Usage:
  # Process variants with pre-computed MSA/templates (from prepare_variant_af3_configs.py)
  python run_alphafold_with_a3m_batch.py \
    --batch_input_dir=example/subtilisin/variants \
    --model_dir=model \
    --run_inference=true

  # Process variants with A3M files (needs template search)
  python run_alphafold_with_a3m_batch.py \
    --batch_input_dir=example/subtilisin/variants \
    --model_dir=model \
    --db_dir=alphafold3_db \
    --run_template_search=true

  # Skip already completed predictions
  python run_alphafold_with_a3m_batch.py \
    --batch_input_dir=variants \
    --model_dir=model \
    --skip_existing=true

Features:
  - Loads model once and reuses for all predictions (faster than running separately)
  - Supports directory scanning for input.json files
  - Automatically detects if MSA/templates are already included in JSON
  - Can skip already completed predictions
  - Supports template search with pre-computed MSAs (when needed)
  - Progress tracking and error handling
"""

import datetime
import glob
import json
import os
import pathlib
import string
import time
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

# Import functions from run_alphafold_with_a3m
from run_alphafold_with_a3m import (
    load_a3m_file,
    clean_a3m_content,
    search_templates_with_msa,
    add_a3m_to_fold_input,
)


FLAGS = flags.FLAGS

# Input options
flags.DEFINE_string(
    'batch_input_dir',
    None,
    'Directory containing subdirectories with input.json files. '
    'Each subdirectory should contain an input.json file.',
)
flags.DEFINE_list(
    'batch_input_dirs',
    None,
    'Comma-separated list of directories containing input.json files. '
    'Alternative to --batch_input_dir for processing specific variants.',
)
flags.DEFINE_string(
    'input_json_name',
    'input.json',
    'Name of the input JSON file to look for in each directory.',
)

# Output options
flags.DEFINE_bool(
    'skip_existing',
    False,
    'Skip predictions if output files already exist.',
)
flags.DEFINE_string(
    'output_marker',
    'model_output.cif',
    'File to check for determining if prediction is complete.',
)

# Note: run_template_search and max_template_date flags are inherited from run_alphafold_with_a3m


def find_input_dirs(base_dir: str, input_json_name: str) -> list[str]:
    """Find all directories containing input.json files."""
    input_dirs = []

    # Check if base_dir itself contains input.json
    if os.path.isfile(os.path.join(base_dir, input_json_name)):
        input_dirs.append(base_dir)

    # Check subdirectories
    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            json_path = os.path.join(item_path, input_json_name)
            if os.path.isfile(json_path):
                input_dirs.append(item_path)

    return input_dirs


def check_input_has_inline_data(input_dict: dict) -> tuple[bool, bool, bool]:
    """Check if input JSON already has inline MSA and templates.

    Returns:
        Tuple of (has_inline_msa, has_inline_templates, has_msa_path)
        - has_inline_msa: True if unpairedMsa is present and non-empty
        - has_inline_templates: True if templates list is present
        - has_msa_path: True if msa_path reference exists (needs loading)
    """
    has_inline_msa = False
    has_inline_templates = False
    has_msa_path = False

    for seq in input_dict.get('sequences', []):
        if 'protein' in seq:
            protein = seq['protein']
            # Check for inline MSA
            if 'unpairedMsa' in protein and protein['unpairedMsa']:
                has_inline_msa = True
            # Check for templates
            if 'templates' in protein:
                has_inline_templates = True
            # Check for msa_path reference
            if 'msa_path' in protein:
                has_msa_path = True

    return has_inline_msa, has_inline_templates, has_msa_path


def is_prediction_complete(output_dir: str, output_marker: str) -> bool:
    """Check if prediction output files exist."""
    # Check for the marker file (with any seed suffix)
    pattern = os.path.join(output_dir, f'*_{output_marker}')
    matches = glob.glob(pattern)
    if matches:
        return True

    # Also check without prefix
    marker_path = os.path.join(output_dir, output_marker)
    return os.path.isfile(marker_path)


def process_single_input(
    input_json_path: str,
    output_dir: str,
    model_runner: run_alphafold.ModelRunner | None,
    buckets: list[int] | None,
    run_template_search: bool,
    max_template_date: datetime.date | None,
    seqres_database_path: str | None,
    pdb_database_path: str | None,
    hmmsearch_binary_path: str | None,
    hmmbuild_binary_path: str | None,
) -> bool:
    """Process a single input.json file.

    This function automatically detects the input format:
    1. If input JSON already contains unpairedMsa and templates (from wt_data_json),
       it skips MSA/template search and runs inference directly.
    2. If input JSON contains msa_path references, it loads the A3M files and
       optionally searches for templates.

    Returns:
        True if successful, False if failed.
    """
    input_json_dir = os.path.dirname(os.path.abspath(input_json_path))

    # Load input JSON
    with open(input_json_path, 'r') as f:
        input_json = f.read()

    # Check input format
    input_dict = json.loads(input_json)
    has_inline_msa, has_inline_templates, has_msa_path = check_input_has_inline_data(input_dict)

    # Determine processing mode
    if has_inline_msa and has_inline_templates:
        # Input already has MSA and templates - run inference directly
        logging.info('  Input has inline MSA and templates - skipping search')
    elif has_msa_path:
        # Input has msa_path references - load A3M files and optionally search templates
        logging.info('  Input has msa_path - loading A3M files')
        input_json = add_a3m_to_fold_input(
            input_json,
            unpaired_msa_path=None,
            unpaired_msa_dir=None,
            input_json_dir=input_json_dir,
            run_template_search=run_template_search,
            max_template_date=max_template_date,
            seqres_database_path=seqres_database_path,
            pdb_database_path=pdb_database_path,
            hmmsearch_binary_path=hmmsearch_binary_path,
            hmmbuild_binary_path=hmmbuild_binary_path,
        )
    elif has_inline_msa and not has_inline_templates:
        # Has MSA but no templates - optionally search for templates
        if run_template_search:
            logging.info('  Input has inline MSA but no templates - searching templates')
            input_json = add_a3m_to_fold_input(
                input_json,
                unpaired_msa_path=None,
                unpaired_msa_dir=None,
                input_json_dir=input_json_dir,
                run_template_search=run_template_search,
                max_template_date=max_template_date,
                seqres_database_path=seqres_database_path,
                pdb_database_path=pdb_database_path,
                hmmsearch_binary_path=hmmsearch_binary_path,
                hmmbuild_binary_path=hmmbuild_binary_path,
            )
        else:
            logging.info('  Input has inline MSA, no templates - using as is')
    else:
        # No MSA source - this may fail unless the data pipeline is run
        logging.warning('  No MSA source found in input JSON')

    # Parse fold input
    fold_input = folding_input.Input.from_json(input_json)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run AlphaFold 3
    results = run_alphafold.process_fold_input(
        fold_input=fold_input,
        data_pipeline_config=None,
        model_runner=model_runner,
        output_dir=output_dir,
        buckets=buckets,
    )

    return results is not None


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Validate input options
    if FLAGS.batch_input_dir is None and FLAGS.batch_input_dirs is None:
        raise app.UsageError('Either --batch_input_dir or --batch_input_dirs must be specified.')
    if FLAGS.batch_input_dir is not None and FLAGS.batch_input_dirs is not None:
        raise app.UsageError('Only one of --batch_input_dir or --batch_input_dirs can be specified.')

    # Find all input directories
    if FLAGS.batch_input_dir:
        input_dirs = find_input_dirs(FLAGS.batch_input_dir, FLAGS.input_json_name)
    else:
        input_dirs = [d.strip() for d in FLAGS.batch_input_dirs]

    if not input_dirs:
        raise app.UsageError(f'No input directories found with {FLAGS.input_json_name}')

    logging.info(f'Found {len(input_dirs)} input directories to process')

    # Filter out completed predictions if requested
    if FLAGS.skip_existing:
        pending_dirs = []
        skipped_count = 0
        for input_dir in input_dirs:
            if is_prediction_complete(input_dir, FLAGS.output_marker):
                skipped_count += 1
            else:
                pending_dirs.append(input_dir)

        if skipped_count > 0:
            logging.info(f'Skipping {skipped_count} already completed predictions')
        input_dirs = pending_dirs

    if not input_dirs:
        logging.info('All predictions already complete. Nothing to do.')
        return

    logging.info(f'Will process {len(input_dirs)} predictions')

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

    # Setup model runner (load once, reuse for all predictions)
    model_runner = None
    if FLAGS.run_inference:
        logging.info('Setting up model runner (this may take a moment)...')
        model_runner = run_alphafold.ModelRunner(
            model_class=run_alphafold.diffusion_model.Diffuser,
            config=run_alphafold.make_model_config(),
            device=jax.local_devices()[0],
            model_dir=pathlib.Path(FLAGS.model_dir),
        )
        logging.info('Model runner ready')

    # Parse buckets
    buckets = None
    if FLAGS.buckets:
        buckets = [int(b) for b in FLAGS.buckets]

    # Process each input
    successful = 0
    failed = 0
    failed_dirs = []

    total = len(input_dirs)
    for i, input_dir in enumerate(input_dirs, 1):
        input_json_path = os.path.join(input_dir, FLAGS.input_json_name)
        variant_name = os.path.basename(input_dir)

        logging.info(f'[{i}/{total}] Processing {variant_name}...')
        start_time = time.time()

        try:
            success = process_single_input(
                input_json_path=input_json_path,
                output_dir=input_dir,
                model_runner=model_runner,
                buckets=buckets,
                run_template_search=FLAGS.run_template_search,
                max_template_date=max_template_date,
                seqres_database_path=seqres_database_path,
                pdb_database_path=pdb_database_path,
                hmmsearch_binary_path=FLAGS.hmmsearch_binary_path,
                hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
            )

            elapsed = time.time() - start_time
            if success:
                successful += 1
                logging.info(f'[{i}/{total}] Completed {variant_name} in {elapsed:.1f}s')
            else:
                failed += 1
                failed_dirs.append(input_dir)
                logging.warning(f'[{i}/{total}] Failed {variant_name} (no results)')

        except Exception as e:
            elapsed = time.time() - start_time
            failed += 1
            failed_dirs.append(input_dir)
            logging.error(f'[{i}/{total}] Error processing {variant_name}: {e}')

    # Summary
    logging.info('=' * 60)
    logging.info(f'Batch processing complete')
    logging.info(f'  Successful: {successful}/{total}')
    logging.info(f'  Failed: {failed}/{total}')

    if failed_dirs:
        logging.info('Failed directories:')
        for d in failed_dirs:
            logging.info(f'  - {d}')


if __name__ == '__main__':
    app.run(main)
