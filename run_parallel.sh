
#!/bin/bash

mkdir -p logs_sv_parallel


datasets=("tictactoe" "adult" "dota2")
attack_methods=("random" "fedavg" "arima" "moirai")
client_nums=(2 4 6 8)
declare -A alpha_map
alpha_map[tictactoe]="10 100 1000 20000"
alpha_map[adult]="1000 100000 1000000 20000000"
alpha_map[dota2]="10000 1000000 50000000 200000000"


MAX_PARALLEL=4

function wait_for_jobs {
  while [ "$(jobs | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 5
  done
}

for trial_id in {0..1}; do
  for dataset in "${datasets[@]}"; do
    for attack_method in "${attack_methods[@]}"; do
      for client_num in "${client_nums[@]}"; do
        for alpha in ${alpha_map[$dataset]}; do

          # SV only
          script="attack_exp_final_auto_sv.py"
          log_dir="logs_sv_parallel"

          ## non-attacking
          wait_for_jobs
          echo "Launching: [SV] dataset=$dataset, attack=$attack_method, alpha=$alpha, N=$client_num, trial=$trial_id, use_attack=False"
          nohup python $script \
            --dataset $dataset \
            --attack_method $attack_method \
            --client_num $client_num \
            --alpha $alpha \
            --attacker_id 0 \
            --contribution_method shapley \
            --trial_id $trial_id \
            > $log_dir/${dataset}_${attack_method}_${alpha}_${client_num}_${trial_id}_noattack.log 2>&1 &

          # attacking
          for ((attacker_id=0; attacker_id<client_num; attacker_id++)); do
            wait_for_jobs
            echo "Launching: [SV] dataset=$dataset, attack=$attack_method, alpha=$alpha, N=$client_num, trial=$trial_id, attacker=$attacker_id"
            nohup python $script \
              --dataset $dataset \
              --attack_method $attack_method \
              --client_num $client_num \
              --alpha $alpha \
              --attacker_id $attacker_id \
              --contribution_method shapley \
              --trial_id $trial_id \
              --use_attack \
              > $log_dir/${dataset}_${attack_method}_${alpha}_${client_num}_${trial_id}_attacker${attacker_id}.log 2>&1 &
          done

        done
      done
    done
  done
done
