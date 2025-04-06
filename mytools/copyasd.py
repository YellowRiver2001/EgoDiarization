import os
import shutil
import glob

def copy_matched_files(sourcebase, targetbase):
    """
    Copy speaker_global_EGO4d.txt files from ../asd directory to matching EgoDiarization/exp/test_poweset subdirectories
    """
    # Source and target directories
    source_base = "dataset/asd"
    target_base = "exp/test1/mid"
    
    # Target file name to copy
    target_file = "speaker_global_EGO4d.txt"
    
    # Ensure target base directory exists
    if not os.path.exists(target_base):
        print(f"Target directory {target_base} does not exist")
        return
    
    # Get all subdirectories in target directory
    target_subdirs = [d for d in os.listdir(target_base) 
                     if os.path.isdir(os.path.join(target_base, d))]
    
    # Get all subdirectories in source directory
    if not os.path.exists(source_base):
        print(f"Source directory {source_base} does not exist")
        return
        
    source_subdirs = [d for d in os.listdir(source_base) 
                      if os.path.isdir(os.path.join(source_base, d))]
    
    # Record operation results
    copied_count = 0
    missing_source_files = []
    missing_target_dirs = []
    
    # For each target subdirectory, check if there's a matching source subdirectory
    for subdir in target_subdirs:
        if subdir in source_subdirs:
            source_file = os.path.join(source_base, subdir, target_file)
            target_dir = os.path.join(target_base, subdir)
            target_file_path = os.path.join(target_dir, target_file)
            
            # Check if source file exists
            if os.path.exists(source_file):
                # Copy file
                shutil.copy2(source_file, target_file_path)
                print(f"Copied: {source_file} -> {target_file_path}")
                copied_count += 1
            else:
                missing_source_files.append(subdir)
                print(f"Warning: Source file does not exist {source_file}")
        else:
            missing_target_dirs.append(subdir)
            print(f"Warning: No matching subdirectory found in source directory {subdir}")
    
    # Print summary
    print("\n=== Copy Operation Completed ===")
    print(f"Successfully copied: {copied_count} files")
    print(f"Missing source files: {len(missing_source_files)} directories")
    print(f"No matching items in source directory: {len(missing_target_dirs)} directories")
    
    if missing_source_files:
        print("\nDirectories missing source files:")
        for d in missing_source_files:
            print(f"- {d}")
    
    if missing_target_dirs:
        print("\nDirectories with no matching items in source directory:")
        for d in missing_target_dirs[:10]:  # Print only the first 10 to avoid excessive output
            print(f"- {d}")
        if len(missing_target_dirs) > 10:
            print(f"... and {len(missing_target_dirs) - 10} other directories")

if __name__ == "__main__":
    copy_matched_files()