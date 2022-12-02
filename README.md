# Setup

```
conda create --name trime python=3.7.11
conda activate trime
pip install --upgrade pip
export TMPDIR=/data/local/tmp/  # Annoying pip install [Errno 28] No space left on device
pip install --editable .
pip install torch==1.9.1+cu111  -f https://download.pytorch.org/whl/cu111/torch_stable.html  --no-cache-dir
conda install -c pytorch faiss-gpu
conda deactivate
```

# Toy Data

```
python preprocess.py --only-source --trainpref data-bin/toy/raw_data/train.txt --validpref data-bin/toy/raw_data/val.txt --testpref data-bin/toy/raw_data/test.txt --destdir data-bin/toy --workers 2
```


# Commands

[Google Sheet](https://docs.google.com/spreadsheets/d/1IRtkVaqOn9s7LEn0Agqrhkgy-wzY5wcvIwpX_tKDFLw/edit#gid=1165240396)

```
python main.py /tmp/model data/toy/ --max_length 37 --batch_size 7 --model_name gpt2 --epochs 3 --gpu 0
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py scratch/model data/wikitext-2-raw-v1 --batch_size 1 --model_name gpt2 --epochs 3 --gpus 0,1,2,3,4,5,6,7 --lr 0.1
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py scratch/model data/wikitext-2-raw-v1 --batch_size 1 --model_name gpt2-medium --epochs 3 --gpus 0,1,2,3,4,5,6,7 --lr 0.1
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py scratch/model data/wikitext-2-raw-v1 --batch_size 1 --model_name gpt2-large --epochs 3 --gpus 0,1,2,3,4,5,6,7 --lr 0.1
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py scratch/model data/wikitext-2-raw-v1 --batch_size 1 --model_name gpt2-xl --epochs 3 --gpus 0,1,2,3,4,5,6,7 --lr 0.1
```
