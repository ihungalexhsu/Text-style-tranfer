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
#beta = ['10']
gamma = ['100', '10', '1']
#gamma = ['10']
delta = ['10','100','1000']
#delta = ['100','10']
eta = ['10','100']
configs = ['config/config_proposed_regularize.yaml']
# command template
command_template = 'python main.py -m style_regularize -c {} --test --train --alpha 1 --beta {} --gamma {} --delta {} --zeta {} --eta {}'
counter = 0
for idx,config in enumerate(configs):
    for b in beta:
        for g in gamma:
            for e in eta:
                for d in delta:
                    z = d
                    command = command_template.format(config,b,g,d,z,e)
                    bash_file = 'regularize_scripts/run_proposed_regularize-b{}-g{}-d{}-z{}-e{}.sh'.format(b,g,d,z,e)
                    with open( bash_file, 'w' ) as OUT:
                        OUT.write('source ~/.zshrc\n')
                        OUT.write('rm -rf ~/.nv\n')
                        OUT.write('cd ~/Code/Text-style-tranfer-with-style-embedding-contraint\n')
                        OUT.write(command)
                    if counter < 100: 
                        qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h=\'!vista13&!vista05&!vista11&!vista06\',h_rt=24:00:00,h_vmem=8G -pe mt 8 {}'.format(bash_file, bash_file)
                    else:
                        qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h=\'!vista13&!vista05&!vista11\',h_rt=24:00:00,h_vmem=8G,gpu=1 -q ephemeral.q {}'.format(bash_file, bash_file)
                    counter+=1
                    os.system( qsub_command )
                    print( qsub_command )
                    print( 'Submitted' )
