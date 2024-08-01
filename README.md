# +

baseline for our exp

#### env and run

##### env-gpushare

```sh
curl -#OL "https://download.gpushare.com/download/update_source"
chmod u+x ./update_source
./update_source apt
curl -#OL "https://download.gpushare.com/download/update_source"
chmod u+x ./update_source
./update_source conda
```

##### env-conda

```sh
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

##### env-docker

```sh
# python -m pip install --upgrade pip wheel setuptools
```

##### check

```sh
nvidia-smi
which python
python -c 'import torch;print(torch.cuda.is_available())'
lscpu
lsmem
lspci
```

#### dev

```sh
# python -m pip install seaborn pillow matplotlib tqdm pandas
# python -m pip install transformers[torch]
# python -m pip install pytorch_transformers
# python -m pip install torchrec-nightly torchtext numba
```

##### run

```sh
screen -R cvfl
cd cvfl/ModelNet_CVFL
python quant_cifar.py 10class/classes/ --num_clients 4 --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 8 --vecdim 2 --comp quantize
Ctrl A D
```
