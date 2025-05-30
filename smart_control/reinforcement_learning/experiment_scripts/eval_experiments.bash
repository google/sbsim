# python scripts/eval.py --policy-dir experiment_results/ddpg_train_run-july-6th_2025_04_07-12:50:40/policies/ --gin-config /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-11-06.gin --experiment-name ddpg_train-summer_eval-winter
# python scripts/eval.py --policy-dir schedule --gin-config /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-11-06.gin --experiment-name schedule_eval-winter

# python scripts/eval.py --policy-dir experiment_results/sac_multiple_episodes_2025_04_27-15:36:58/policies/ --gin-config /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-10-21.gin --experiment-name sac_multiple_episodes_eval-10-21
# python scripts/eval.py --policy-dir schedule --gin-config /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-10-21.gin --experiment-name schedule_eval-10-21

# python scripts/eval.py --policy-dir experiment_results/ddpg_train_run-july-6th_2025_04_07-12:50:40/policies/ --gin-config /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-10-21.gin --experiment-name ddpg_train-summer_eval-10-21

parallel -j 4 --progress <<EOF
python scripts/eval.py --policy-dir eval_results/ddpg_train-summer_eval-08-06_2025_04_15-01:44:58/trajectories/episode_0.json --gin-config /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-08-06.gin --aggregate 168 --experiment-name aggregate-168_ddpg_train-summer_eval-08-06
python scripts/eval.py --policy-dir eval_results/ddpg_train-summer_eval-09-06_2025_04_14-23:54:58/trajectories/episode_0.json --gin-config /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-09-06.gin --aggregate 168 --experiment-name aggregate-168_ddpg_train-summer_eval-09-06
python scripts/eval.py --policy-dir eval_results/ddpg_train-summer_eval-10-06_2025_04_14-21:49:08/trajectories/episode_0.json --gin-config /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-10-06.gin --aggregate 168 --experiment-name aggregate-168_ddpg_train-summer_eval-10-06
python scripts/eval.py --policy-dir eval_results/ddpg_train-summer_eval-winter_2025_04_14-12:25:39/trajectories/episode_0.json --gin-config /home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-11-06.gin --aggregate 168 --experiment-name aggregate-168_ddpg_train-summer_eval-11-06
EOF

echo "All evaluations completed."
