source ~/.zshrc
cd ~/Code/pytorch-style-transfer-aaai18
python main.py -m seq2seq -c config/config_beta5.yaml --train --test --load_model