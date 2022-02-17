# --use_aug means using data augmentation for training (translation, rotation, flip)
# --is_braid means traing the model on braid hairstyles with shape loss (smoothing loss)
# If you want to train the models from scratch, then remove `--continue_train` and `--epoch` command
python train.py --dataroot ./dataset/braid --name S2I_braid --netG unet_at_bg --model pix2pix_hair --dataset_mode hair --use_aug --is_braid --batch_size 4 --save_epoch_freq 50 --epoch_count 1 --n_epochs 400 --n_epochs_decay 0 --display_freq 20 --save_latest_freq 10000 --print_freq 100 --no_flip --gpu_ids 1 --continue_train --epoch 400 
