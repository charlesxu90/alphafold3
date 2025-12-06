# PY3=/home/xux/miniforge3/envs/af3-env/bin/python
# AF3_path=/home/xux/Desktop/done_projects/Struct_pred/0.general_complex/alphafold3

AF3_path=/home/xux/Desktop/ProteinMCP/ProteinMCP/mcp-servers/alphafold3_mcp/repo/alphafold3
PY3=/home/xux/Desktop/ProteinMCP/ProteinMCP/mcp-servers/alphafold3_mcp/env/bin/python
# Run alphafold3
data_path=$1
# set device to 0 if $2 is not set
if [ -z "$2" ]; then
	device=0
else
	device=$2
fi

for input_json in `ls $data_path/*/input.json`; do
	XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=$device \
		$PY3 $AF3_path/run_alphafold.py \
		--json_path=$input_json \
		--model_dir=$AF3_path/model \
		--db_dir=$AF3_path/alphafold3_db \
		--output_dir=$data_path \
		--run_data_pipeline=true --run_inference=true
		# --flash_attention_implementation=xla # need if GPU is not A100
done

# Batch converting cif to pdb using maxit
# for i in `ls $data_path/*/*.cif`; do echo $i; maxit -o 2 -input $i -output $i.pdb; done