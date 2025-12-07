#!/bin/bash
# Batch AlphaFold3 prediction script for protein variants
# Usage: ./run_af3_local_batch.sh <input_dir> [device]
# Example: ./run_af3_local_batch.sh example/subtilisin/variants 0
#
# The script automatically detects the input format:
# - If input.json contains inline unpairedMsa and templates (from prepare_variant_af3_configs.py
#   with --wt_data_json), it skips MSA/template search and runs inference directly.
# - If input.json contains msa_path references, it loads A3M files and searches templates.

PY3=/home/xux/miniforge3/envs/af3-env/bin/python
AF3_path=/home/xux/Desktop/done_projects/Struct_pred/0.general_complex/alphafold3

# Parse arguments
input_dir=$1
if [ -z "$input_dir" ]; then
    echo "Usage: $0 <input_dir> [device]"
    echo "  input_dir: Directory containing variant subdirectories with input.json files"
    echo "  device: GPU device number (default: 0)"
    echo ""
    echo "The script auto-detects input format:"
    echo "  - Inline MSA/templates: runs inference directly (fast)"
    echo "  - msa_path references: loads A3M and searches templates"
    exit 1
fi

# Set device (default: 0)
if [ -z "$2" ]; then
    device=0
else
    device=$2
fi

echo "Running batch AlphaFold3 predictions"
echo "  Input directory: $input_dir"
echo "  GPU device: $device"
echo ""

XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=$device \
    $PY3 $AF3_path/run_alphafold_with_a3m_batch.py \
    --batch_input_dir=$input_dir \
    --model_dir=$AF3_path/model \
    --db_dir=$AF3_path/alphafold3_db \
    --run_template_search=true \
    --skip_existing=true \
    --run_inference=true
    # --flash_attention_implementation=xla  # Use if GPU is not A100
    # Note: --run_template_search is only used when input has msa_path references.
    #       Inputs with inline MSA/templates skip the search automatically.

# Optional: Batch convert cif to pdb
# for variant_dir in $input_dir/*/; do
#     for cif in $variant_dir/*.cif; do
#         [ -f "$cif" ] && maxit -input $cif -output ${cif%.cif}.pdb -o 2
#     done
# done
# rm maxit.log