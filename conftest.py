"""pytest 根配置: 把 scripts/ 加入 import 路径。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
