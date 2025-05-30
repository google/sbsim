python scripts/train.py \
    --starter-buffer-path data/starter_buffers/default_starter_buffer_seqlen2_exp6720/2025-04-04T06\:30\:49.808661634-04\:00/ \
    --experiment-name sac_multiple_episodes \
    --scenario-config-path "/home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-07-06.gin" \
                           "/home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-08-06.gin" \
                           "/home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-10-06.gin" \
                           "/home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-11-06.gin" \
    --eval-scenario-config-path "/home/gabriel-user/projects/sbsim/smart_control/configs/resources/sb1/generated_configs/config_timestepsec-900_numdaysinepisode-14_starttimestamp-2023-10-21.gin" 
