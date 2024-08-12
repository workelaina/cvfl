# +

baseline for our exp

#### env and run

##### env-gpushare

```sh
curl -#OL "https://download.gpushare.com/download/update_source"
chmod u+x ./update_source
./update_source apt
# 7 (bfsu)
curl -#OL "https://download.gpushare.com/download/update_source"
chmod u+x ./update_source
./update_source conda
# 1 (bfsu)
apt update
apt upgrade
apt install screen tree pciutils pkg-config
vim ~/.ssh/authorized_keys
```

```md
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC+3l60NdskymyIypkG9Asr3jFFy/KFCF3dTJzd9AJ11iIb8t18KZHmZ2lNv4GcAPp0y+bZcFFT42uJa8rjvcJxIQbOru61YTHSpXeCOm6/j2er7RtBp1aE4Jfk2K4QO++A35FJSwkNYaxn+qkfJj6MZHV0JpX8Zgyuvz8oKIXjlmcLAeN4VSja6+IM9ht0q7LQK/zaqtP+5ek88CPXbFvNcLCJ0Zyg5f+gI/liFFAvZjTmDp8EG3i6VJL40xPEH5/Q8b8ppkXgvufZ5o4s6iY1ssnCGVS/viGm/MxOkmmXl3xv4p56i9DnbIZwEHMM1EIlFtkCAR7nIvY3gQvaSBDLeDPrO3gDsA8Fa3JxAvea0/2Bp/L7mam6jdS89QT/rGyqAroXGEANJDkLY6GKDxVH+3dUyD3y7MgW2SuMuCe6c3yphbk/wNdDNB0UvZ7zoCScWV69EfcrcwCPWbdKM9jb3JfCUkPsXUAy7+lTSLxXvdC36bzxmwI3or3YS8oxDq+J1argf/k44YwJNPNuTmyGiPxiMU4PFuKOjch9mtXOGN6Oti+dCZY1SzyECEFtqgJwH/jnodRcxG7UEcsHU4gY3JZBfenG4J12SJICZzyEbdsJZeq7SAeEx5OwDRqdNFOGueIwtwEbajQ2Eg1K9DgIQfDUW+gulfa6XGKl/8MgMQ== userelaina@pm.me
```

```sh
vim ~/.bashrc
```

```sh
git config --global http.sslVerify "false"
alias py='python'
export TMOUT=600
cd /hy-tmp/
conda activate py37
```

##### env-conda

```sh
conda create -y -n py37 python=3.7
conda activate py37
which python
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
pip install --upgrade pip wheel setuptools -i https://mirrors.jlu.edu.cn/pypi/simple
pip install torchvision==0.5 -i https://mirrors.jlu.edu.cn/pypi/simple
pip install scikit-learn tqdm scipy pandas matplotlib -i https://mirrors.jlu.edu.cn/pypi/simple
pip install latbin -i https://mirrors.jlu.edu.cn/pypi/simple
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

##### run

```sh
# git clone https://github.com/workelaina/cvfl.git
scp -P 45583 train.py root@i-2.gpushare.com:/hy-tmp/
screen -R cvfl
python train.py 10class/classes/ --num_clients 2 --b 128 --local_epochs 2 --epochs 150 --lr 0.0001 --quant_level 8 --vecdim 2 --comp quantize
Ctrl A D
```
