import os
import shutil

# 定义源文件夹和目标文件夹的根路径
source_root = "/home/rx/audio-visual/Light-ASD/demo"
target_root = "/home/rx/audio-visual/dbavsd/EgoDiarization/exp/test1/mid"

# 需要转移的文件和文件夹名称
files_to_move = [
    "audio.wav",
    # "clean_nosplit_segments.txt",
    # "split_wav_powerset_Ego4d_segments",
    # "clean_wav_powerset_Ego4d_segments"
]

def move_files():
    # 遍历源文件夹下的所有子文件夹
    for folder_name in os.listdir(source_root):
        source_folder_path = os.path.join(source_root, folder_name)
        
        # 检查是否是目录并且名称符合预期格式（这里简单检查是否为目录）
        if os.path.isdir(source_folder_path):
            # 构建目标文件夹路径
            target_folder_path = os.path.join(target_root, folder_name)
            
            # 如果目标文件夹不存在，创建它
            if not os.path.exists(target_folder_path):
                continue
            
            # 转移每个指定的文件/文件夹
            for item in files_to_move:
                source_item_path = os.path.join(source_folder_path, "pyavi", item)
                target_item_path = os.path.join(target_folder_path, item)
                
                # 检查源文件/文件夹是否存在
                if os.path.exists(source_item_path):
                    try:
                        # 如果目标路径已存在，先删除
                        if os.path.exists(target_item_path):
                            if os.path.isdir(target_item_path):
                                shutil.rmtree(target_item_path)
                            else:
                                os.remove(target_item_path)
                        
# 根据是文件还是文件夹选择复制方式
                        if os.path.isfile(source_item_path):
                            shutil.copy2(source_item_path, target_item_path)
                            print(f"成功复制文件: {source_item_path} -> {target_item_path}")
                        elif os.path.isdir(source_item_path):
                            shutil.copytree(source_item_path, target_item_path)
                            print(f"成功复制文件夹: {source_item_path} -> {target_item_path}")
                    except Exception as e:
                        print(f"复制失败 {source_item_path}: {str(e)}")
                else:
                    print(f"源文件不存在: {source_item_path}")

if __name__ == "__main__":
    # 执行文件转移
    move_files()