import os
import pickle
import json
from typing import List


def subdirs(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)


def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def check_dir(dir):
    if not os.path.exists(dir):
        mkdir(dir)


def mkdir(path):
    '''
    创建目录
    '''
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    path=path.rstrip("/")
    # 判断路径是否存在
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录,创建目录操作函数
        '''
        os.mkdir(path)与os.makedirs(path)的区别是,当父目录不存在的时候os.mkdir(path)不会创建，os.makedirs(path)则会创建父目录
        '''
        #此处路径最好使用utf-8解码，否则在磁盘中可能会出现乱码的情况
        # os.makedirs(path.decode('utf-8')) 
        os.makedirs(path) 
        print(path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False

# def get_filename_number(filename, num):
#     # 提取文件名中的编号
#     return int(re.findall("\d+",filename)[num])

def get_case_number(filename):

    return int((filename.split('/')[-1].split('_')[1]))

def get_filename_info(filefullname):
    # 提取文件名的文件夹路径、文件名、文件短名称、文件扩展名
    (folderpath,filename) = os.path.split(filefullname)
    (shotname,extension) = os.path.splitext(filename) # 文件名和后缀
    return folderpath, filename, shotname, extension
