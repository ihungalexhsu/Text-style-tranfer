qsub -P other -j y -o scripts/run_proposed.sh.output -cwd -l h=\!vista13\&\!vista05\&\!vista11\&\!vista06,h_rt=24:00:00,h_vmem=10G,gpu=3 -q ephemeral.q scripts/run_proposed.sh
