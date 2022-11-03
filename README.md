# pytorch_chinese_multiple_choice
基于pytorch+lstm的中文多项选择。数据来源CAIL2022中的[司法考试]([CAIL (China AI and Law Challenge) Official website (cipsc.org.cn)](http://cail.cipsc.org.cn/task0.html?raceID=0&cail_tag=2022))任务，部分代码参考[china-ai-law-challenge/CAIL2022 (github.com)](https://github.com/china-ai-law-challenge/CAIL2022)。

# 目录结构

```python
checkpoints：保存模型
config：配置文件
----config_parser.py：解析配置文件
----sfks.config：配置文件
data：数据
----sfks：
--------mid_data：存储词表
--------raw_data：存储原始数据
------------train.json：训练数据
------------test_input.json：要预测的数据
data_loader：数据加载器
logs：日志
models：模型
----encoder：编码器
----layer：存储一些神经网络层
----qa：主模型
test：测试相关模块
utils：辅助函数
cutter.py：分词获取词表
main.py：主运行文件
preds.txt：预测保存的文件
```

# 依赖

```python
pytorch
configparser
```

# 运行

```python
python main.py
```

会进行训练，并保存预测结果到preds.txt里面。
