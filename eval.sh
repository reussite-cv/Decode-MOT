cd ./src
python track.py mot_sche --add_decision_head --add_attention --exp_id mot_sche_tracking --test_mot17 True --load_model ../pretrained/decodemot_mot17_final.pth --conf_thres 0.6 --decision_thres 0.5 --data_dir ../data
cd ..
