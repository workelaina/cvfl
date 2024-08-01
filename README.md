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
apt install screen tree pciutils
vim ~/.ssh/authorized_keys
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
pip install --upgrade pip wheel setuptools
pip install torchvision==0.5.0 -i https://mirrors.jlu.edu.cn/pypi/simple
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
git clone https://github.com/workelaina/cvfl.git
scp -P 60571 -r ModelNet_CVFL/ root@i-1.gpushare.com:/hy-tmp/
screen -R cvfl
cd ModelNet_CVFL
python quant_cifar.py 10class/classes/ --num_clients 4 --b 100 --local_epochs 10 --epochs 200 --lr 0.0001 --quant_level 8 --vecdim 2 --comp quantize
Ctrl A D
```
