for i in ./*.wav; do whisperx "$i" --model BSC-LT/faster-whisper-bsc-large-v3-cat --output_format srt --condition_on_previous_text True --max_line_width 250  --max_line_count 1  --segment_resolution sentence  --align_model WAV2VEC2_ASR_LARGE_LV60K_960H; done
for i in ./*.wav; do echo "$i" ; done

whisperx ./dataset/7767db7ab74c86e79bb141dbb9f31235.wav --model BSC-LT/faster-whisper-bsc-large-v3-cat --output_format srt --condition_on_previous_text True --max_line_width 250  --max_line_count 1  --segment_resolution sentence  --align_model WAV2VEC2_ASR_LARGE_LV60K_960H
whisperx ./dataset/d/1/7/d17dc1f059ebf74545157f24ad7a1e31.wav --model BSC-LT/faster-whisper-bsc-large-v3-cat --output_format srt --condition_on_previous_text True --max_line_width 250  --max_line_count 1  --segment_resolution sentence  --align_model WAV2VEC2_ASR_LARGE_LV60K_960H

whisperx ./dataset/7767db7ab74c86e79bb141dbb9f31235.wav --model BSC-LT/faster-whisper-bsc-large-v3-cat --output_format srt --condition_on_previous_text True --max_line_count 1  --segment_resolution sentence  --align_model WAV2VEC2_ASR_LARGE_LV60K_960H
