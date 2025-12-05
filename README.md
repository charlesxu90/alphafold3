# Install

## Create the environment with the following commands
```shell
mamba env create -f environment.yml

# activate environment
mamba activate af3-env
```

## Install the requirements with pip
```shell
pip install -r requirements.txt
```

## Build and install alphafold3
```shell
# Need to use g++ 11.4 in bulding
sudo mv /usr/bin/g++ /usr/bin/g++.bak
sudo ln -s /usr/bin/g++-11 /usr/bin/g++

pip install -e .

# install maxit
# download maxit source code from https://sw-tools.rcsb.org/apps/MAXIT/source.html
cd maxit-*
make
make binary

# move to /opt and add to PATH
sudo mv maxit-* /opt
export RCSBROOT=/opt/maxit-v11.300-prod-src
export PATH="$RCSBROOT/bin:"$PATH
```

## Fix a bug from JAX
```shell
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

# Usage

## Prepare the Json config file
```json
{
  "name": "2PV7",
  "sequences": [
    {
      "protein": {
        "id": ["A", "B"],
        "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```
Need to give the sequences, seeds. MSA can be searched using AF3 by default.
Note: the `"id"` need to be a single character, e.g. 'A', 'B', or 'C'.

## Run the program
```shell
bash run_af3.sh
```
```shell
# run_af3.sh
PY3=/home/xiaopeng/miniconda3/envs/af3-env/bin/python
source activate /home/xiaopeng/miniconda3/envs/af3-env
AF3_path=/home/xiaopeng/Desktop/Struct_pred/alphafold3

input_json=$AF3_path/output/input.json
# input_json=$AF3_path/output/2pv7/2pv7_data.json
# input_json=$AF3_path/output/8aw3/8aw3_data.json #input_protien_rna_ion.json
#input_json=$AF3_path/output/7bbv/7bbv_data.json #input_protien_gly_ion.json

XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=0 \
	$PY3 run_alphafold.py \
	--json_path=$input_json \
	--model_dir=$AF3_path/model \
	--db_dir=$AF3_path/alphafold3_db \
	--output_dir=$AF3_path/output \
	--run_data_pipeline=true --run_inference=true
	# --flash_attention_implementation=xla # need if GPU is not A100
```
