# Using Pre-computed A3M MSA Files with AlphaFold 3

This guide explains how to use pre-computed A3M MSA (Multiple Sequence Alignment) files with AlphaFold 3, bypassing the MSA search pipeline.

## Overview

AlphaFold 3 typically runs an MSA search pipeline using tools like Jackhmmer, HMMER, etc. However, you may want to use pre-computed MSAs when:

1. **You already have MSAs** from external sources (MMseqs2, HHblits, etc.)
2. **Reusing MSAs** to save computation time
3. **Testing different MSAs** without re-running the search
4. **Using custom MSAs** for specialized scenarios

## A3M File Format

A3M is a compact MSA format similar to FASTA. Key features:

- **Uppercase letters**: Match states (aligned to query)
- **Lowercase letters**: Insert states (insertions relative to query)
- **Dashes (-)**: Deletions in the alignment

### Example A3M File

```
>query_sequence
MKLLVLGLGSLLLLLLLA
>homolog_1/1-18
MKLLVLGLGS--LLLLLL
>homolog_2/1-20
MKLLVLGLGSaaLLLLLL
```

In this example:
- `homolog_1` has a deletion (--) at positions 11-12
- `homolog_2` has insertions (aa) between positions 10 and 11

## Methods for Using A3M Files

### Method 1: Using Helper Functions in `run_alphafold_test.py`

Two helper functions are available:

#### `load_a3m_file(a3m_path: str) -> str`

Loads an A3M file and returns its contents as a string.

```python
from run_alphafold_test import load_a3m_file

unpaired_msa = load_a3m_file('/path/to/unpaired.a3m')
```

#### `create_fold_input_with_a3m(...)`

Creates a fold input JSON with A3M files automatically loaded.

```python
from run_alphafold_test import create_fold_input_with_a3m
from alphafold3.common import folding_input

# Define your sequences
sequences = [
    {
        'protein': {
            'id': 'A',
            'sequence': 'MKLLVLGLGSLLLLLLLA...',
            'modifications': [],
        }
    },
    {
        'protein': {
            'id': 'B',
            'sequence': 'GATVDPTRLL...',
            'modifications': [],
        }
    },
]

# Create fold input with A3M files
input_json = create_fold_input_with_a3m(
    name='my_protein',
    sequences=sequences,
    model_seeds=[1234],
    unpaired_msa_paths={
        'A': '/path/to/chainA_unpaired.a3m',
        'B': '/path/to/chainB_unpaired.a3m',
    },
    paired_msa_paths={
        'A': '/path/to/chainA_paired.a3m',
    },
)

# Parse and use
fold_input = folding_input.Input.from_json(input_json)
```

### Method 2: Using the Standalone Script

The `run_alphafold_with_a3m.py` script provides a complete solution:

#### Basic Usage

```bash
python run_alphafold_with_a3m.py \
  --input_json=input.json \
  --unpaired_msa_a3m=unpaired.a3m \
  --paired_msa_a3m=paired.a3m \
  --output_dir=output/ \
  --model_dir=model/
```

#### Per-Chain A3M Files

For multi-chain complexes, organize A3M files by chain ID:

```
msa_files/
  ├── A.a3m  # Unpaired MSA for chain A
  ├── B.a3m  # Unpaired MSA for chain B
  └── C.a3m  # Unpaired MSA for chain C
```

Then run:

```bash
python run_alphafold_with_a3m.py \
  --input_json=complex.json \
  --unpaired_msa_dir=msa_files/ \
  --output_dir=output/ \
  --model_dir=model/
```

#### Available Options

- `--input_json`: Path to input JSON file (required)
- `--unpaired_msa_a3m`: Single unpaired MSA file (applied to first protein)
- `--paired_msa_a3m`: Single paired MSA file (applied to first protein)
- `--unpaired_msa_dir`: Directory with per-chain unpaired A3M files
- `--paired_msa_dir`: Directory with per-chain paired A3M files
- `--output_dir`: Output directory (required)
- `--model_dir`: Model parameters directory
- `--buckets`: Token bucket sizes (e.g., "256,512,1024")
- `--run_inference`: Whether to run inference (default: true)

### Method 3: Manual JSON Modification

You can manually add MSA data to your input JSON:

```json
{
  "name": "my_protein",
  "modelSeeds": [1234],
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "MKLLVLGLGSLLLLLLLA",
        "unpairedMsa": ">query\nMKLLVLGLGSLLLLLLLA\n>hit1\nMKLLVLGLGS--LLLLLL",
        "pairedMsa": ">query\nMKLLVLGLGSLLLLLLLA\n>paired1\nMKLLVLGLGS--LLLLLL",
        "modifications": []
      }
    }
  ]
}
```

Then run normally without the MSA pipeline:

```python
from alphafold3.common import folding_input
import run_alphafold

# Load input with MSAs
with open('input_with_msa.json', 'r') as f:
    fold_input = folding_input.Input.from_json(f.read())

# Run without data pipeline
results = run_alphafold.process_fold_input(
    fold_input=fold_input,
    data_pipeline_config=None,  # Skip MSA search
    model_runner=model_runner,
    output_dir='output/',
)
```

## Complete Example

Here's a complete workflow:

### 1. Prepare A3M Files

```bash
# Example: Generate A3M with MMseqs2
mmseqs easy-search \
  query.fasta \
  database \
  result.m8 \
  tmp \
  --format-mode 3 \
  -a > unpaired.a3m
```

### 2. Create Input JSON

```json
{
  "name": "5tgy",
  "modelSeeds": [1234],
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "SEFEKLRQTGDELVQAFQRLREIFDKGDDDSLEQVLEEIEELIQKHRQLFDNRQEAADTEAAKQGDQWVQLFQRFREAIDKGDKDSLEQLLEELEQALQKIRELAEKKN",
        "modifications": []
      }
    },
    {
      "ligand": {
        "id": "B",
        "ccdCodes": ["7BU"]
      }
    }
  ]
}
```

### 3. Run with A3M Files

```bash
python run_alphafold_with_a3m.py \
  --input_json=5tgy_input.json \
  --unpaired_msa_a3m=unpaired.a3m \
  --paired_msa_a3m=paired.a3m \
  --output_dir=5tgy_output/ \
  --model_dir=model/
```

## MSA Requirements

### Protein Chains

- **Unpaired MSA**: Required for featurization
- **Paired MSA**: Optional, used for inter-chain contacts
- **Templates**: Optional, can be omitted

### RNA Chains

- **Unpaired MSA**: Required for featurization
- **Paired MSA**: Not used for RNA

### DNA Chains

- MSAs are not typically used for DNA

## Testing

A test case is included in `run_alphafold_test.py`:

```bash
python -m pytest run_alphafold_test.py::InferenceTest::test_inference_with_a3m_files
```

Or with unittest:

```bash
python run_alphafold_test.py InferenceTest.test_inference_with_a3m_files
```

## Troubleshooting

### "missing unpaired MSA" Error

This means the input doesn't have MSA data and no MSA pipeline was configured:

```python
# ❌ Wrong - no MSAs and no pipeline
run_alphafold.process_fold_input(
    fold_input=fold_input,  # No MSAs in input
    data_pipeline_config=None,  # No pipeline
    model_runner=model_runner,
    output_dir='output/',
)

# ✅ Correct - provide MSAs
fold_input_with_msa = ...  # Load A3M files
run_alphafold.process_fold_input(
    fold_input=fold_input_with_msa,
    data_pipeline_config=None,
    model_runner=model_runner,
    output_dir='output/',
)
```

### Empty or Invalid A3M Files

Ensure your A3M files:
- Start with the query sequence
- Use proper FASTA format (>header followed by sequence)
- Don't have extra whitespace or formatting issues

### Performance Considerations

- **MSA depth**: More sequences generally improve accuracy but increase compute
- **MSA quality**: Better-aligned homologs are more valuable than quantity
- **Paired MSAs**: Most beneficial for protein complexes with co-evolving chains

## Additional Resources

- [AlphaFold 3 Documentation](../docs/)
- [Input Format Documentation](../docs/input.md)
- [A3M Format Specification](https://github.com/soedinglab/hh-suite/wiki#the-a3m-format)
- [Example MSA Tools](https://github.com/soedinglab/MMseqs2)

## Summary

You now have three ways to use A3M files with AlphaFold 3:

1. **Helper functions** in `run_alphafold_test.py` for programmatic use
2. **Standalone script** `run_alphafold_with_a3m.py` for command-line use
3. **Manual JSON modification** for complete control

Choose the method that best fits your workflow!
