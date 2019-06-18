
module load Anaconda3/2018.12-rhel-7

export PYTHONPATH="/u/gja/projects/edGNN:$PYTHONPATH"

learning_rates=(0.001)
batch_size=(32)
queue="prod.med"
datasets=("mutagenicity" "ptc_fr" "ptc_fm" "ptc_mm" "ptc_mr")
config_fpath="../core/models/config_files/config_edGNN_graph_class.json"
all_data_path=("../datasets/mutagenicity" "../datasets/ptc_fr" "../datasets/ptc_fm" "../datasets/ptc_mm" "../datasets/ptc_mr")


for i in 0 1 2 3 4
do
	for bs in "${batch_size[@]}"
	do
		for lr in "${learning_rates[@]}"
		do

		    echo "$lr"
		    echo "$bs"
		    echo "$queue"
		    echo "${all_data_path[i]}"
		    echo "$config_fpath"
		    echo "${datasets[i]}"

		    bsub -R "rusage [ngpus_excl_p=1]"  -J  "log" -o "../runs/lsf_logs.%J.stdout" -e "../runs/lsf_logs.%J.stderr" -q "$queue" "python run_experiments --dataset ${datasets[i]} --config_fpath $config_fpath --data_path ${all_data_path[i]} --lr $lr --batch-size $bs --n-epochs 40 --gpu 0 --repeat 10"

			sleep 0.1

		done
	done
done
