import configparser
import functools


class ConfigParser:
    def __init__(self, *args, **params):
        self.config = configparser.RawConfigParser(*args, **params)

    def read(self, filenames, encoding=None):
        self.config.read(filenames, encoding=encoding)


def _build_func(func_name):
    """该函数用于将原有的属性赋给新的自定义的类"""

    @functools.wraps(getattr(configparser.RawConfigParser, func_name))
    def func(self, *args, **kwargs):
        try:
            return getattr(self.config, func_name)(*args, **kwargs)
        except Exception as e:
            print(e)
    return func


def create_config(path):
    """
    config.sections()：获取每一个节点
    config.options("train")：获取指定节点
    config.get("train", "epoch")：获取某节点的某各属性值
    config.items("train")：获取某节点的所有属性值
    config.set("db", "db_port", "69")：修改节点属性值
    config.write(open("ini", "w"))：写入配置
    config.has_section("section")：检查是否存在某节点
    config.remove_section("default")：删除节点
    """
    for func_name in dir(configparser.RawConfigParser):
        if not func_name.startswith('_') and func_name != "read":
            setattr(ConfigParser, func_name, _build_func(func_name))
    config = ConfigParser()
    config.read(path)

    return config
