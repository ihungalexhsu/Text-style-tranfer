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
command_template = 'python main.py -m style_att_adver -c {} --test --train --alpha {} --beta {} --gamma {} --delta {} --zeta {} '
configs = ['config/yelp/config_att_adver.yaml']
# parameters
alpha = ['5', '10', '15']
beta = ['1']
gamma = ['500','1000','1500']
delta = ['0']
zeta = ['1', '10', '20', '50']
for idx,config in enumerate(configs):
    for b in beta:
        for d in delta:
            for g in gamma:
                for a in alpha:
                    for z in zeta:
                        command = command_template.format(config, a, b, g, d, z)
                        bash_file = 'scripts_att_adver/run_proposed-a{}-b{}-g{}-d{}-z{}.sh'.format(a,b,g,d,z)
                        with open( bash_file, 'w' ) as OUT:
                            #OUT.write('rm -rf ~/.nv\n')
                            OUT.write('source ~/.zshrc\n')
                            OUT.write('cd ~/Code/Text-style-transfer-ihung\n')
                            OUT.write(command)
                        qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h=\'!vista13&!vista04&!vista11&!vista05&!vista06&!vista08&!vista20&!vista03\',h_rt=24:00:00,h_vmem=4.5G,gpu=1 -q ephemeral.q -pe mt 2 {}'.format(bash_file, bash_file)
                        os.system( qsub_command )
                        print( qsub_command )
                        print( 'Submitted' )
