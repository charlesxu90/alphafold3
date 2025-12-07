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
