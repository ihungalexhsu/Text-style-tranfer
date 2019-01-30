source ~/.zshrc
cd ~/Code/Robust-Speech-Recognition
python main.py -m seq2seq -c config/config_beta5.yaml --train --test --load_model