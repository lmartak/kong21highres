# Pre-trained model checkpoint files
## Kong https://zenodo.org/records/4034264
## Edwards https://zenodo.org/records/10610212

# TEST on single file

## Kong
python pytorch/inference.py --model_name=kong --model_type=Note_pedal --checkpoint_path='ckpts/kong_note_pedal.pth' --audio_path='resources/BuiJL06M.wav' --cuda

## Edwards
python pytorch/inference.py --model_name=edwards --model_type=Regress_onset_offset_frame_velocity_CRNN --checkpoint_path='ckpts/edwards_regress_onset.pth' --audio_path='resources/BuiJL06M.wav' --cuda

# Transcribe set of files

## MAESTRO/test original recordings (both models, GPU 0)
for wav_path in `ls ~/work/mds24/data/mae/3_maetest_mae/*.wav`; do
    CUDA_VISIBLE_DEVICES=0 python pytorch/inference.py \
        --model_name=kong --model_type=Note_pedal --checkpoint_path='ckpts/kong_note_pedal.pth' \
        --audio_path="$wav_path" --cuda
    CUDA_VISIBLE_DEVICES=0 python pytorch/inference.py \
        --model_name=edwards --model_type=Regress_onset_offset_frame_velocity_CRNN --checkpoint_path='ckpts/edwards_regress_onset.pth' \
        --audio_path="$wav_path" --cuda
done

## MAESTRO/test disklavier recordings (both models, GPU 1)
for wav_path in `ls ~/work/mds24/data/dk/3_maetest_dk/*.wav`; do
    CUDA_VISIBLE_DEVICES=1 python pytorch/inference.py \
        --model_name=kong --model_type=Note_pedal --checkpoint_path='ckpts/kong_note_pedal.pth' \
        --audio_path="$wav_path" --cuda
    CUDA_VISIBLE_DEVICES=1 python pytorch/inference.py \
        --model_name=edwards --model_type=Regress_onset_offset_frame_velocity_CRNN --checkpoint_path='ckpts/edwards_regress_onset.pth' \
        --audio_path="$wav_path" --cuda
done
