source ~/.zshrc
rm -rf ~/.nv
cd ~/Code/Text-style-tranfer-with-style-embedding-contraint
python main.py -m style_regularize -c config/config_proposed_regularize.yaml --test --train --alpha 1 --beta 10 --gamma 10 --delta 100 --zeta 100 --eta 100