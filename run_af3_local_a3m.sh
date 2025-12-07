# PY3=/home/xux/miniforge3/envs/af3-env/bin/python
# AF3_path=/home/xux/Desktop/done_projects/Struct_pred/0.general_complex/alphafold3
AF3_path=$(pwd)
# Run alphafold3
data_path=$1
# set device to 0 if $2 is not set
if [ -z "$2" ]; then
	device=0
else
	device=$2
fi

# 1iep
# data_path=/mnt/data/done_projects/Struct_pred/0.general_complex/alphafold3/example/1iep_a3m_fix
input_json=$data_path/input.json
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=$device \
	python $AF3_path/run_alphafold_with_a3m.py \
	--input_json=$input_json \
	--model_dir=$AF3_path/model \
	--db_dir=$AF3_path/alphafold3_db \
	--output_dir=$data_path \
	--run_template_search=true \
	--run_inference=true
	# --run_template_search=false  # Use this to skip template search (faster)
	# --flash_attention_implementation=xla # need if GPU is not A100

# Batch converting cif to pdb
# for i in $data_path/*.cif; do maxit -input $i -output $i.pdb -o 2; done 
# rm maxit.log
