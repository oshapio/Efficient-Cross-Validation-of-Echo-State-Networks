#!/bin/bash

cd precision_experiments/time_series_output

for seed in {1..5}; do
    echo "Running output_exps with seed - $seed"
    {
        (python output_exp.py --valid_method cross --final_model retrain --seed $seed &&
         python output_exp.py --valid_method cross --final_model average --seed $seed &&
         python output_exp.py --valid_method cross --final_model best --seed $seed) &

         (python output_exp.py --valid_method accum --final_model retrain --seed $seed &&
          python output_exp.py --valid_method accum --final_model average --seed $seed &&
          python output_exp.py --valid_method accum --final_model best --seed $seed) &

         (python output_exp.py --valid_method walk_forward --final_model retrain --seed $seed &&
          python output_exp.py --valid_method walk_forward --final_model average --seed $seed &&
          python output_exp.py --valid_method walk_forward --final_model best --seed $seed) &

         (python output_exp.py --valid_method standard --final_model retrain --seed $seed &&
          python output_exp.py --valid_method standard --final_model average --seed $seed)
    }
    wait
done
