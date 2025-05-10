import json
def save_json_to_file(data, file_path='data.json'):
    """
    将 JSON 数据保存到文件中。

    Args:
        data (dict): 要保存的 JSON 数据。
        file_path (str, optional): 要保存的文件路径，默认为 'data.json'。

    Returns:
        bool: 数据保存成功返回 True，失败返回 False。
    """
    try:
        # 检查数据是否为字典类型
        if not isinstance(data, dict):
            print("Error: Data is not a JSON object.")
            return False

        # 打开文件并写入数据
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"Data successfully saved to {file_path}")
        return True
    except Exception as e:
        print(f"An error occurred while saving data: {e}")
        return False

import json

def load_json_file(file_path):
    """
    从 JSON 文件中加载数据。

    Args:
        file_path (str): JSON 文件的路径。

    Returns:
        dict: 文件中的 JSON 数据转换为字典格式。
             如果读取失败或文件不存在，则返回 None。
    """
    try:
        # 打开并读取文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None