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
command_template = 'python main.py -m style_proposed -c {} --test --train --alpha {} --beta 1 --gamma {} --delta 5 --load_model'
configs = ['config/yelp/config_proposed.yaml']
# parameters
gamma = ['50', '100', '500', '1000', '2000']
alpha = ['1', '10', '100', '0.1']
for idx,config in enumerate(configs):
    for g in gamma:
        for a in alpha:
            command = command_template.format(config, a, g)
            bash_file = 'scripts_proposed/run_proposed-a{}-g{}.sh'.format(a,g)
            with open( bash_file, 'w' ) as OUT:
                OUT.write('sleep 3\n')
                #OUT.write('rm -rf ~/.nv\n')
                OUT.write('source ~/.zshrc\n')
                OUT.write('cd ~/Code/Text-style-transfer-ihung\n')
                OUT.write(command)
            qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h=\'!vista13&!vista03&!vista11\',h_rt=24:00:00,h_vmem=4.5G,gpu=2 -q ephemeral.q -pe mt 4 {}'.format(bash_file, bash_file)
            os.system( qsub_command )
            print( qsub_command )
            print( 'Submitted' )
'''
accept_pair = [(0.1,100),(0.1,50),
               (1,100),(1,1000),
               (1,500),(10,100),
               (10,50),(10,500),
               (100,100),(100,1000),
               (100,50),(100,500)]
for ap in accept_pair:
    a = str(ap[0])
    g = str(ap[1])
    config = configs[0]
    command = command_template.format(config, a, g)
    bash_file = 'scripts_proposed/run_proposed-a{}-g{}.sh'.format(a,g)
    with open( bash_file, 'w' ) as OUT:
        OUT.write('source ~/.zshrc\n')
        OUT.write('rm -rf ~/.nv\n')
        OUT.write('cd ~/Code/Text-style-transfer-ihung\n')
        OUT.write(command)
    qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h=\'!vista13&!vista03&!vista11\',h_rt=24:00:00,h_vmem=4.5G,gpu=2 -q ephemeral.q -pe mt 4 {}'.format(bash_file, bash_file)
    os.system( qsub_command )
    print( qsub_command )
    print( 'Submitted' )
'''
