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
styles = ['0.1', '0.5', '1', '5', '10', '15', '20', '25', '30','50','100']
#styles = ['1','25']
configs = ['config/yelp/config_fader.yaml']
# command template
command_template = 'python main.py -m style_fader -c {} --test --load_model --alpha {} --load_model'
for idx,config in enumerate(configs):
    for s in styles:
        command = command_template.format(config, s)
        bash_file = 'test_scripts/run_fader-s{}.sh'.format(s)
        with open( bash_file, 'w' ) as OUT:
            OUT.write('source ~/.zshrc\n')
            OUT.write('cd ~/Code/Text-style-transfer-ihung\n')
            OUT.write(command)
        qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h=\'!vista13&!vista03\',h_rt=24:00:00,h_vmem=4.5G,gpu=1 -q ephemeral.q -pe mt 4 {}'.format(bash_file, bash_file)
        os.system( qsub_command )
        print( qsub_command )
        print( 'Submitted' )
