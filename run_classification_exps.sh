#!/bin/bash

cd precision_experiments/time_series_classification

for seed in {6..1000}; do
    echo "Running output_exps with seed - $seed"
    {
        (python classification_exp.py --valid_method cross --final_model retrain --reg default --seed $seed &
        python classification_exp.py --valid_method cross --final_model retrain --reg individ --seed $seed &
        python classification_exp.py --valid_method cross --final_model average --reg default --seed $seed)
    }
    wait

    {
        (python classification_exp.py --valid_method cross --final_model average --reg individ --seed $seed &
	python classification_exp.py --valid_method standard --final_model retrain --reg default --seed $seed &
        python classification_exp.py --valid_method standard --final_model average --reg default --seed $seed)
    }
    wait
done
