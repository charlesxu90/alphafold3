PY3=/home/xux/miniconda3/envs/af3-env/bin/python
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/xux/miniconda3/envs/af3-env
AF3_path=/home/xux/Desktop/done_projects/Prot_pred/ref_works/0.struct_pred/alphafold3

# Run alphafold3
data_path=$1
# set device to 0 if $2 is not set
if [ -z "$2" ]; then
	device=0
else
	device=$2
fi

for input_json in `ls $data_path/*/data.json`; do
	XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=$device \
		$PY3 $AF3_path/run_alphafold.py \
		--json_path=$input_json \
		--model_dir=$AF3_path/model \
		--db_dir=$AF3_path/alphafold3_db \
		--output_dir=$data_path \
		--run_data_pipeline=false --run_inference=true
		# --flash_attention_implementation=xla # need if GPU is not A100
done

# Batch converting cif to pdb
# Using maxit
# maxit -o 2 -input /home/xux/Desktop/BestzymeP/Glucoamylase/Gluco_opt/data/BSJ_data_processed/Yufan_R5_DD_CD.csv-af3/1404-m291/1404-m291_model.cif  -output /home/xux/Desktop/BestzymeP/Glucoamylase/Gluco_opt/data/BSJ_data_processed/Yufan_R5_DD_CD.csv-af3/1404-m291/1404-m291_model.cif.pdb
for i in `ls $data_path/*/*.cif`; do echo $i; maxit -o 2 -input $i -output $i.pdb; done
# Using openbabel
# for i in `ls $data_path/*/*.cif`; do echo $i; obabel -i cif $i -o pdb -O $i.pdb -p "PerceiveBondOrders"; done 
# obabel -i cif results/savinase/top40_lasa_R4.csv-af3/m103/m103_model.cif -o pdb -O results/savinase/top40_lasa_R4.csv-af3/m103/m103_model.cif.pdb