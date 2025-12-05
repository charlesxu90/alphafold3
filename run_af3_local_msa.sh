PY3=/home/xux/miniconda3/envs/af3-env/bin/python
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/xux/miniconda3/envs/af3-env
AF3_path=/home/xux/Desktop/done_projects/Struct_pred/0.general_complex/alphafold3

# Phytase WT
# input_json=$AF3_path/output/phytase_phytic_acid.json
# input_json=$AF3_path/output/phytase_wt/phytase_phytic_acid_data.json
# input_json=$AF3_path/output/phytase_phytic_acid/phytase_phytic_acid_data.json

# Phytase L32
# data_path=/mnt/data/done_projects/Prot_pred/ref_works/0.struct_pred/alphafold3/results/phytase
# input_json=$data_path/l32/data.json
# input_json=$data_path/l32_2p/data.json
# input_json=$data_path/l32_3p/data.json
# input_json=$data_path/l32_4p/data.json
# input_json=$data_path/l32_5p/data.json
# input_json=$data_path/l32_5p_ccd/data.json
# input_json=$data_path/l32_6p/data.json

# data_path=/mnt/data/done_projects/Prot_pred/ref_works/0.struct_pred/alphafold3/results/savinase
# data_path=/mnt/data/done_projects/Prot_pred/ref_works/0.struct_pred/alphafold3/results/glucoamylase
# input_json=$data_path/input.json

# Savinase_wt
# data_path=/mnt/data/done_projects/Prot_pred/ref_works/0.struct_pred/alphafold3/results/savinase/ca2_AAPF
# input_json=$data_path/wt/data.json

# PH20
# data_path=/mnt/data/done_projects/Prot_pred/ref_works/0.struct_pred/alphafold3/results/PH20/
# input_json=$data_path/input.json

# TaqP
# data_path=/mnt/data/done_projects/Prot_pred/ref_works/0.struct_pred/alphafold3/results/taqp/
# input_json=$data_path/input.json

# Savi 880
# data_path=/mnt/data/done_projects/Struct_pred/0.general_complex/alphafold3/results/savinase/880/
# input_json=$data_path/input.json
# --db_dir=/mnt/data/done_projects/Struct_pred/0.general_complex/alphafold3/results/savinase/880/savi_db \

# ACE
# data_path=/mnt/data/done_projects/Struct_pred/0.general_complex/alphafold3/results/ace/
# input_json=$data_path/input.json
data_path=/mnt/data/done_projects/Struct_pred/0.general_complex/alphafold3/results/ace/
# input_json=$data_path/ace_agt1/data.json
input_json=$data_path/ace_capt/data.json

# tdt
#data_path=/mnt/data/done_projects/Struct_pred/0.general_complex/alphafold3/results/tdt/zatdt/
#input_json=$data_path/input.json

# 1iep
data_path=/mnt/data/done_projects/Struct_pred/0.general_complex/alphafold3/results/1iep/
input_json=$data_path/input.json
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=0 \
	$PY3 run_alphafold.py \
	--json_path=$input_json \
	--model_dir=$AF3_path/model \
	--db_dir=$AF3_path/alphafold3_db \
	--output_dir=$data_path \
	--run_data_pipeline=true --run_inference=true
	# --flash_attention_implementation=xla # need if GPU is not A100

# Batch converting cif to pdb
for i in $data_path/*.cif; do maxit -input $i -output $i.pdb -o 2; done 

