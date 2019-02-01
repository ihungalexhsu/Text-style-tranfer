source ~/.zshrc
cd ~/Code/pytorch-style-transfer-aaai18
python main.py -m seq2seq -c config/config_beta10.yaml --train --test 
