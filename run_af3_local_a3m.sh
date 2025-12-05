source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/xux/miniforge3/envs/af3-env
AF3_path=/home/xux/Desktop/done_projects/Struct_pred/0.general_complex/alphafold3

# tdt
data_path=results/tdt/variants_cf/af3/test/
input_json=$data_path/input.json
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=1 \
	python run_alphafold_with_a3m.py \
	--input_json=$input_json \
	--model_dir=$AF3_path/model \
	--unpaired_msa_dir=results/tdt/variants_cf/boltz2/cah2322405.1/boltz_results_input_config/msa/input_config_unpaired_tmp_env/ \
	--db_dir=$AF3_path/alphafold3_db \
	--output_dir=$data_path \
	# --flash_attention_implementation=xla # need if GPU is not A100

# Batch converting cif to pdb
# for i in $data_path/*.cif; do maxit -input $i -output $i.pdb -o 2; done 

