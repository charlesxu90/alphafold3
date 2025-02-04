PY3=/home/xux/miniconda3/envs/af3-env/bin/python
source activate /home/xux/miniconda3/envs/af3-env
AF3_path=/home/xux/Desktop/Struct_pred/alphafold3

# Run alphafold3
data_path=$1

for input_json in `ls $data_path/*/data.json`; do
	XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=0 \
		$PY3 run_alphafold.py \
		--json_path=$input_json \
		--model_dir=$AF3_path/model \
		--db_dir=$AF3_path/alphafold3_db \
		--output_dir=$data_path \
		--run_data_pipeline=false --run_inference=true
		# --flash_attention_implementation=xla # need if GPU is not A100
done

# Batch converting cif to pdb
for i in `ls $data_path/*/*.cif`; do maxit -input $i -output $i.pdb -o 2; done 
