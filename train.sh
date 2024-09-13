# adapt SAM to RGB+NIR modality with lora PEFT
net=sam_lora
modality=rgbnir
lr=3e-4
proj_type=simmat
exp_name=${modality}_${proj_type}_${net}_lr${lr}

python -u train.py -net ${net} \
  -proj_type ${proj_type} \
  -exp_name ${exp_name} \
  -lr ${lr} \
  -b 4 -modality ${modality} -val_freq 5
