import os

# --- Configuration ---
# The fast directory where your denoised/filtered files live
CLEANED_AUDIO_DIR = "/leonardo_scratch/large/userexternal/sgaliano/wavs_cleaned"

# Paths to your original dirty filelists
ORIGINAL_TRAIN = "Data/train_list.txt"
ORIGINAL_VAL = "Data/val_list.txt"

# Paths where the new filtered filelists will be saved
NEW_TRAIN = "Data/train_list_clean.txt"
NEW_VAL = "Data/val_list_clean.txt"

def update_list(original_path, new_path):
    if not os.path.exists(original_path):
        print(f"Error: Original list path not found at: {original_path}")
        return 0, 0

    kept_count = 0
    dropped_count = 0

    with open(original_path, "r", encoding="utf-8") as infile, \
         open(new_path, "w", encoding="utf-8") as outfile:
        
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            
            # Split based on your exact format: wavs/file.wav|phonemes|speaker_id
            parts = line.split("|")
            if len(parts) != 3:
                print(f"Warning [Line {line_num}]: Malformed sequence skipped -> {line}")
                continue
                
            old_wav_path, phonemes, speaker_id = parts
            
            # Extract only the explicit filename (e.g., '777f2ac1c7f88090b5c5c5264b3e5afc.wav')
            filename = os.path.basename(old_wav_path)
            
            # Verify if this file survived the NISQA filter and exists in wavs_cleaned
            target_file_path = os.path.join(CLEANED_AUDIO_DIR, filename)
            
            if os.path.exists(target_file_path):
                # Reconstruct the line preserving your original formatting structure
                # Writing 'wavs/filename.wav' allows StyleTTS to append it cleanly to root_path
                new_wav_path = f"wavs/{filename}"
                outfile.write(f"{new_wav_path}|{phonemes}|{speaker_id}\n")
                kept_count += 1
            else:
                dropped_count += 1

    return kept_count, dropped_count

if __name__ == "__main__":
    print("=== Launching StyleTTS Dataset Alignment Engine ===")
    
    # 1. Processing train file list
    print(f"\n[1/2] Filtering Train List: {ORIGINAL_TRAIN}...")
    t_kept, t_drop = update_list(ORIGINAL_TRAIN, NEW_TRAIN)
    print(f" -> Successfully synced. Kept: {t_kept} | Dropped: {t_drop}")

    # 2. Processing validation file list
    print(f"\n[2/2] Filtering Validation List: {ORIGINAL_VAL}...")
    v_kept, v_drop = update_list(ORIGINAL_VAL, NEW_VAL)
    print(f" -> Successfully synced. Kept: {v_kept} | Dropped: {v_drop}")

    print("\n=== Realignment Complete ===")
    print(f"New train filelist created at: {NEW_TRAIN}")
    print(f"New validation filelist created at: {NEW_VAL}")