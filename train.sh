cd ./src
python train.py mot_sche --freeze_detector --add_decision_head --add_attention --gpus 1 --num_workers 0 --batch_size 6 --lr 0.00005 --exp_id DecodeMOT --load_model ../pretrained/mot17only_default_model_final.pth --data_cfg lib/cfg/custom/mot17_exceptMOT17-09.json --decision_dim 64 --buffer_size 4
cd ..
