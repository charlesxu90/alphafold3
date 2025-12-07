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
data_path=/mnt/data/done_projects/Struct_pred/0.general_complex/alphafold3/example/1iep_msa_fix
input_json=$data_path/input.json
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=$device \
	python $AF3_path/run_alphafold.py \
	--json_path=$input_json \
	--model_dir=$AF3_path/model \
	--db_dir=$AF3_path/alphafold3_db \
	--output_dir=$data_path \
	--run_data_pipeline=false --run_inference=true
	# --flash_attention_implementation=xla # need if GPU is not A100

# Batch converting cif to pdb
for i in $data_path/*.cif; do maxit -input $i -output $i.pdb -o 2; done 
rm maxit.log
