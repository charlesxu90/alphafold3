PY3=/home/xiaopeng/miniconda3/envs/af3-env/bin/python
source activate /home/xiaopeng/miniconda3/envs/af3-env
AF3_path=/home/xiaopeng/Desktop/Struct_pred/alphafold3

# Phytase WT
# input_json=$AF3_path/output/phytase_phytic_acid.json
# input_json=$AF3_path/output/phytase_wt/phytase_phytic_acid_data.json
# input_json=$AF3_path/output/phytase_phytic_acid/phytase_phytic_acid_data.json

# Phytase L32
input_json=$AF3_path/output/L32_phytic_acid/L32_phytic_acid_data.json

XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=0 \
	$PY3 run_alphafold.py \
	--json_path=$input_json \
	--model_dir=$AF3_path/model \
	--db_dir=$AF3_path/alphafold3_db \
	--output_dir=$AF3_path/output \
	--run_data_pipeline=false --run_inference=true
	# --flash_attention_implementation=xla # need if GPU is not A100
