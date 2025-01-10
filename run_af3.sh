# !!! 目前参数版本是内部的，不可以做宣传，不可以发论文，请知悉!!!
# To work around a known XLA issue causing the compilation time to greatly
# increase, the following environment variable setting XLA flags must be enabled
# when running AlphaFold 3:
# ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

# Memory settings used for folding up to 5,120 tokens on A100 80 GB.
# ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
# ENV XLA_CLIENT_MEM_FRACTION=0.95

PY3=/ibex/user/zhac/software/miniconda3/envs/af3/bin/python

#input_json=/ibex/scratch/projects/c2108/zhac/public_soft/alphafold3/output/input.json
input_json=/ibex/scratch/projects/c2108/zhac/public_soft/alphafold3/output/2pv7/2pv7_data.json
# input_json=/ibex/scratch/projects/c2108/zhac/public_soft/alphafold3/output/8aw3/8aw3_data.json #input_protien_rna_ion.json
#input_json=/ibex/scratch/projects/c2108/zhac/public_soft/alphafold3/output/7bbv/7bbv_data.json #input_protien_gly_ion.json

source activate /ibex/user/zhac/software/miniconda3/envs/af3
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=0 \
	$PY3 run_alphafold.py \
	--json_path=$input_json \
	--model_dir=/ibex/scratch/projects/c2108/zhac/public_soft/alphafold3/model \
	--db_dir=/ibex/scratch/projects/c2108/zhac/public_data/alphafold3_db \
	--output_dir=/ibex/scratch/projects/c2108/zhac/public_soft/alphafold3/output \
	--run_data_pipeline=false --run_inference=true
	# --flash_attention_implementation=xla # need if GPU is not A100
