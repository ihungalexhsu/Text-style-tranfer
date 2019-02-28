'''
script for submitting jobs to runjob.
your script to run on runjob should be input several paramters by using argparser
This script will generate a bash file which is submitted to runjob. this bash file will run your actual work script
output file will located in runjob_outputs/xxx.output
'''
from __future__ import print_function
import os
from itertools import product
from time import sleep
# command template
command_template = 'python main.py -m style_proposed -c {} --test --train --alpha {} --beta 1 --gamma {} --delta 0 --load_model'
configs = ['config/yelp/config_proposed.yaml']
# parameters
gamma = ['0.1','1','5', '10', '20', '50', '100','200']
alpha = ['0.1', '1', '5', '10']
for idx,config in enumerate(configs):
    for g in gamma:
        for a in alpha:
            command = command_template.format(config, a, g)
            bash_file = 'scripts_proposed/run_proposed-a{}-g{}.sh'.format(a,g)
            with open( bash_file, 'w' ) as OUT:
                #OUT.write('rm -rf ~/.nv\n')
                OUT.write('source ~/.zshrc\n')
                OUT.write('cd ~/Code/Text-style-transfer-ihung\n')
                OUT.write(command)
            qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h=\'!vista13&!vista04&!vista11&!vista05&!vista06&!vista08&!vista20&!vista03\',h_rt=24:00:00,h_vmem=4.5G,gpu=1 -q ephemeral.q -pe mt 2 {}'.format(bash_file, bash_file)
            os.system( qsub_command )
            print( qsub_command )
            print( 'Submitted' )
