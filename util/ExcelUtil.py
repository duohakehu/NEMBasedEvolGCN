# @Author : CuiGuonan
# @Time: 2022-12-20
# @File: ExcelUtil.py
import json

import numpy as np
import pandas as pd


class ExcelUtil:
    @staticmethod
    def write_to_Excel(filename, content):
        writer = pd.ExcelWriter(filename)
        df = pd.DataFrame(content)
        df.to_excel(writer, index=False)
        writer.close()

    @staticmethod
    def write_to_Excel_from_dict(filename, content):
        writer = pd.ExcelWriter(filename)
        df = pd.DataFrame.from_dict(content, orient='index')
        df.to_excel(writer, index=False)
        writer.close()

    @staticmethod
    def read_to_Excel(filename):
        data = pd.read_excel(filename)
        return data

    @staticmethod
    def write_to_byte(filename, content):
        np.save(filename, content)

    @staticmethod
    def read_to_byte(filename):
        return np.load(filename)

    @staticmethod
    def write_to_Json(filename, content):
        with open(filename, 'w') as json_file:
            json.dump(content, json_file, indent=4)
