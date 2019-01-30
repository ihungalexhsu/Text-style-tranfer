'''
script for submitting jobs to runjob.
your script to run on runjob should be input several paramters by using argparser
This script will generate a bash file which is submitted to runjob. this bash file will run your actual work script
output file will located in runjob_outputs/xxx.output
'''
from __future__ import print_function
import os
from itertools import product

# parameters
#beta = ['1', '0.1', '0.01', '0.001', '0.0001']
#gamma = ['10', '1', '0.1']
configs = ['config/config_beta1.yaml','config/config_beta5.yaml','config/config_beta10.yaml']
# command template
command_template = 'python main.py -m seq2seq -c {} --train --test '

for idx,config in enumerate(configs):
    command = command_template.format(config)
    if idx==0:
        bash_file = 'runs/naive_style_transfer-beta1.sh'
        with open( bash_file, 'w' ) as OUT:
            OUT.write('source ~/.zshrc\n')
            OUT.write('cd ~/Code/Robust-Speech-Recognition\n')
            OUT.write(command+'--load_model')
    if idx==1:
        bash_file = 'runs/naive_style_transfer-beta5.sh'
        with open( bash_file, 'w' ) as OUT:
            OUT.write('source ~/.zshrc\n')
            OUT.write('cd ~/Code/Robust-Speech-Recognition\n')
            OUT.write(command+'--load_model')
    if idx==2:
        bash_file = 'runs/naive_style_transfer-beta10.sh'
        with open( bash_file, 'w' ) as OUT:
            OUT.write('source ~/.zshrc\n')
            OUT.write('cd ~/Code/Robust-Speech-Recognition\n')
            OUT.write(command)
    if idx!=2:
        qsub_command = 'qsub -P other -j y -o runs_output/{}.output -cwd -l h=\'!vista13\',h_rt=24:00:00,h_vmem=8G,gpu=2 -q ephemeral.q {}'.format(bash_file, bash_file)
    else:
        qsub_command = 'qsub -P other -j y -o runs_output/{}.output -cwd -l h=\'!vista13\',h_rt=24:00:00,h_vmem=8G,gpu=4 -q ephemeral.q {}'.format(bash_file, bash_file)
    os.system( qsub_command )
    print( qsub_command )
    print( 'Submitted' )
