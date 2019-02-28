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
command_template = 'python main.py -m style_att_adver -c {} --test --train --alpha {} --beta {} --gamma {} --delta {} --zeta {} --load_model'
configs = ['config/yelp/config_proposed_att_adver.yaml']
# parameters
alpha = ['0.1','0.5','1','5','10','20']
beta = ['1']
gamma = ['0.1','1','5','10','20','50','100','200']
delta = ['0']
#zeta = ['1', '10', '20', '50']
existing_pair=[(1.,1.,10.,0.,2.),
               (1.,1.,100.,0.,20.),
               (1.,1.,200.,0.,40.),
               (1.,1.,50.,0.,10.),
               (10.,1.,10.,0.,2.),
               (10.,1.,200.,0.,40.),
               (20.,1.,200.,0.,40.),
               (20.,1.,50.,0.,10.),
               (20.,1.,500.,0.,100.),
               (50.,1.,100.,0.,20.),
               (50.,1.,200.,0.,40.),
               (50.,1.,50.,0.,10.),
               (50.,1.,500.,0.,100.)]
for idx,config in enumerate(configs):
    for b in beta:
        for d in delta:
            for a in alpha:
                for g in gamma:
                    z = g
                    #if (float(a),float(b),float(g),float(d),float(z)) in existing_pair:
                    #    print("pass the param")
                    #else:
                    command = command_template.format(config, a, b, g, d, z)
                    bash_file = 'scripts_att_adver/run_proposed-a{}-b{}-g{}-d{}-z{}.sh'.format(a,b,g,d,z)
                    with open( bash_file, 'w' ) as OUT:
                        #OUT.write('rm -rf ~/.nv\n')
                        OUT.write('source ~/.zshrc\n')
                        OUT.write('cd ~/Code/Text-style-transfer-ihung\n')
                        OUT.write(command)
                    qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h=\'!vista13&!vista04&!vista11&!vista05&!vista06&!vista08&!vista20&!vista03\',h_rt=24:00:00,h_vmem=4.5G,gpu=1 -pe mt 2 {}'.format(bash_file, bash_file)
                    os.system( qsub_command )
                    print( qsub_command )
                    print( 'Submitted' )
