from pathlib import Path

def ensure_file_directory(file_path):
    """
    
    """
    dir_path = Path(file_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def print_dict(dict):
    for key, value in dict.items():
        print(key, value)