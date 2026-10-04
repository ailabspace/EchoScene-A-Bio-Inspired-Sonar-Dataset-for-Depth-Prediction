REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
V2E=$REPO/third_party/Visual2Echo
DATA=${DATA:-$REPO/data/board}
WORK=${WORK:-$REPO/work}
mkdir -p $WORK/latents $WORK/teacher $WORK/logs
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PY=${PY:-python3}
MINC=${MINC:-$V2E/checkpoints_pretrained/material_pre_trained_minc.pth}

DATA_FLAGS=(--dataset biosonar --img_path $DATA --audio_path $DATA
  --batvision_img_size 128 --depth_resize_method nearest --max_depth 8
  --audio_crop_pre_roll 0.002 --audio_length 0.075
  --audio_nfft 512 --audio_win_length 128 --audio_hop_length 64 --log_spectrogram --audio_normalize
  --audio_bandpass_lo 20000 --audio_bandpass_hi 100000 --audio_butter_order 4)

TRAIN_FLAGS=("${DATA_FLAGS[@]}"
  --backbone Resnet18 --decoder_spatial_entry --audio_norm_type groupnorm --audio_norm_groups 8
  --use_specaugment --batchSize 32 --niter 30 --nThreads 8 --deterministic
  --epoch_save_freq 10 --display_freq 100 --validation_on --validation_freq 385
  --val_include_test --val_split_n 4874 --early_stop_patience 0
  --freeze_nets --depth_loss_type silog --lambda_depth 1.0 --lambda_grad 0.25 --lambda_ssim 0.5
  --lambda_feat_std 1.0 --feat_std_gamma 0.5 --audio_decoder_dropout 0.2
  --lr_audio 5e-4 --weight_decay 0 --warmup_steps 0 --cosine_T_max 11520 --ema_decay 0 --val_max_batches 0
  --lambda_mat 0 --lambda_ccl_mat 0 --lambda_ct 0 --ct_am_scale 0 --init_material_weight $MINC)

TEACHER=${TEACHER:-$WORK/teacher/moge2_vits_ft/model.pt}
MOGE_FLAGS=(--biosonar_rgb_dir $DATA --rgb_teacher moge_v2 --moge_model_id $TEACHER --moge_resolution_level 0 --moge_num_tokens 256 --moge_use_fp16)
