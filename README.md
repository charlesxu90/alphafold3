# AlphaFold3 code based
A self-maintained AlphaFold3 codebase for easily calling and using AlphaFold3.

## Install

### Create conda environment
```shell
mamba env create -f environment.yml
mamba activate af3-env
pip install -r requirements.txt
```

### Build and install alphafold3
```shell
# Need to use g++ 11.4 in bulding
sudo mv /usr/bin/g++ /usr/bin/g++.bak
sudo ln -s /usr/bin/g++-11 /usr/bin/g++

pip install -e .

# Build data
build_data

## Fix a bug from JAX
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Download AlphaFold3 database
bash fetch_databases.sh ./alphafold3_db
```
### Install maxit to convert cif to pdb

```shell
# download maxit source code from https://sw-tools.rcsb.org/apps/MAXIT/source.html
cd maxit-*
make
make binary

# move to /opt and add to PATH
sudo mv maxit-* /opt
export RCSBROOT=/opt/maxit-v11.300-prod-src
export PATH="$RCSBROOT/bin:"$PATH
```

### Install hmmer
```shell
export AF3_REPO_PATH=$(pwd)

mkdir /tmp/hmmer_build
wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /tmp/hmmer_build
cd /tmp/hmmer_build && echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz" | sha256sum --check
tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz
cp $AF3_REPO_PATH/docker/jackhmmer_seq_limit.patch /tmp/hmmer_build
patch -p0 < jackhmmer_seq_limit.patch
cd hmmer-3.4 
./configure --prefix=$CONDA_PREFIX
make -j
make install
cd easel && make install
rm -R /tmp/hmmer_build
``` 

### Download models
```shell
# Obtain license from Google DeepMind, download and put it into $AF3_REPO_PATH/model
```

## Basic usage

### Prepare the Json config file
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

### Run AlphaFold3
```shell
mamba activate /home/xiaopeng/miniconda3/envs/af3-env
AF3_path=${pwd}

input_json=$AF3_path/output/input.json

XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=0 \
	python run_alphafold.py \
	--json_path=$input_json \
	--model_dir=$AF3_path/model \
	--db_dir=$AF3_path/alphafold3_db \
	--output_dir=$AF3_path/output \
	--run_data_pipeline=true --run_inference=true
	# --flash_attention_implementation=xla # need if GPU is not A100
```

## Run AF3 with prepared a3m file

### Config file
```json
{
  "name": "1iep_a3m_fix",
  "sequences": [
    {
      "protein": {
        "id": ["A"],
        "sequence": "MDPSSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVSAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQ",
        "msa_path": "1iep.a3m"
      }
    },
    {
      "ligand": {
        "id": ["B"],
        "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CC[NH+](C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}

```

### Call ALphaFold3
```shell
PY3=~/miniforge3/envs/af3-env/bin/python
AF3_path=~/Desktop/done_projects/Struct_pred/0.general_complex/alphafold3

# 1iep
data_path=/mnt/data/done_projects/Struct_pred/0.general_complex/alphafold3/example/1iep_a3m_fix
input_json=$data_path/input.json
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=0 \
	$PY3 $AF3_path/run_alphafold_with_a3m.py \
	--input_json=$input_json \
	--model_dir=$AF3_path/model \
	--db_dir=$AF3_path/alphafold3_db \
	--output_dir=$data_path \
	--run_template_search=true \
	--run_inference=true
```

## Call AF3 for protein variants

First, predict the structure for wild-type useing the commands above. The batch processing for variants using the commands below.

### Prepare variant input configs
```shell
python prepare_variant_af3_configs.py --variants_fasta=example/subtilisin/sequences.fasta --wt_data_json=example/subtilisin/wt/subtilisin_wt_data.json --output_dir=example/subtilisin/variants
```

### Batch structure prediction for variants
```shell
PY3=~/miniforge3/envs/af3-env/bin/python
AF3_path=~/Desktop/done_projects/Struct_pred/0.general_complex/alphafold3

XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=$device \
    $PY3 $AF3_path/run_alphafold_with_a3m_batch.py \
    --batch_input_dir=$input_dir \
    --model_dir=$AF3_path/model \
    --db_dir=$AF3_path/alphafold3_db \
    --run_template_search=true \
    --skip_existing=true \
    --run_inference=true
```

