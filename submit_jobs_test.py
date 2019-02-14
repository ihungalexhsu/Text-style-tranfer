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
beta = ['10','1']
gamma = ['10', '1', '0.1']
delta = ['100','10','1000']
#zeta = ['100','10','1000']
configs = ['config/config_proposed.yaml']
# command template
command_template = 'python main.py -m style_proposed -c {} --test --load_model --alpha 1 --beta {} --gamma {} --delta {} --zeta {}'
for idx,config in enumerate(configs):
    for b in beta:
        for g in gamma:
            for d in delta:
                z = str(int(d)*5)
                command = command_template.format(config,b,g,d,z)
                bash_file = 'test_scripts/run_proposed-b{}-g{}-d{}-z{}.sh'.format(b,g,d,z)
                with open( bash_file, 'w' ) as OUT:
                    OUT.write('source ~/.zshrc\n')
                    OUT.write('rm -rf ~/.nv\n')
                    OUT.write('cd ~/Code/Text-style-tranfer-with-style-embedding-contraint\n')
                    OUT.write(command)
                qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h=\'!vista13&!vista05&!vista11&!vista06\',h_rt=24:00:00,h_vmem=8.5G,gpu=1 -pe mt 2 {}'.format(bash_file, bash_file)
                os.system( qsub_command )
                print( qsub_command )
                print( 'Submitted' )
