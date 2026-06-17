#nohup python3 main.py -algo FedMoSA -lr 0.0001 -m mosa -mn mosa -nc 5 -data liver_seg -pr 0 -pv 3 -t 4 -go experiment -gpu 0 > FedSAM_liver_5_mosa_1.log 2>&1 &
#nohup python3 main.py -algo FedMoSA -lr 0.0001 -m mosa -mn mosa -nc 5 -data liver_seg -pr 0 -pv 1 -t 2 -go experiment -gpu 0 > FedSAM_liver_5_mosa_2.log 2>&1 &
#nohup python3 main.py -algo FedMoSA -lr 0.0001 -m mosa -mn mosa -nc 4 -data brain_seg -pr 0 -pv 4 -t 5 -go experiment -gpu 0 > FedSAM_brain_4_mosa_3.log 2>&1 &

#nohup python3 main.py -algo FedMSA -lr 0.0001 -m msavanilla -mn msavanilla -nc 5 -data liver_seg -pr 0 -pv 3 -t 4 -go experiment -gpu 0 > FedSAM_liver_5_samvanilla_1.log 2>&1 &
#nohup python3 main.py -algo FedMSA -lr 0.0001 -m msavanilla -mn msavanilla -nc 5 -data liver_seg -pr 0 -pv 1 -t 2 -go experiment -gpu 0 > FedSAM_liver_5_samvanilla_2.log 2>&1 &
#nohup python3 main.py -algo FedMSA -lr 0.0001 -m msavanilla -mn msavanilla -nc 5 -data liver_seg -pr 0 -pv 4 -t 5 -go experiment -gpu 0 > FedSAM_liver_5_samvanilla_3.log 2>&1 &

#nohup python3 main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -pr 0 -pv 3 -t 4 -go experiment -gpu 0 > FednnUNET_liver_5_nnunet_1.log 2>&1 &
#nohup python3 main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -pr 0 -pv 1 -t 2 -go experiment -gpu 0 > FednnUNET_liver_5_nnunet_2.log 2>&1 &
#nohup python3 main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -pr 0 -pv 4 -t 5 -go experiment -gpu 0 > FednnUNET_liver_5_nnunet_3.log 2>&1 &

#nohup python3 main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -afa 1 -pr 0 -pv 3 -t 4 -go experiment-afa -gpu 0 > FednnUNET_liver_5_nnunet_1_AFA.log 2>&1 &
#nohup python3 main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -afa 1 -pr 0 -pv 1 -t 2 -go experiment-afa -gpu 0 > FednnUNET_liver_5_nnunet_2_AFA.log 2>&1 &
#nohup python3 main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -afa 1 -pr 0 -pv 4 -t 5 -go experiment-afa -gpu 0 > FednnUNET_liver_5_nnunet_3_AFA.log 2>&1 &
