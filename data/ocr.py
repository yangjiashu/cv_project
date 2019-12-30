from aip import AipOcr

APP_ID = '18104709'
API_KEY = '9mzaSjALI2BOGYcXwNgN6Yru'
SECRET_KEY = 'E1TWlIfgkhyVflXaXBYvW156A8iqHrgG'

aipOcr  = AipOcr(APP_ID, API_KEY, SECRET_KEY)

# 读取图片
filePath = "results/003/1.jpg"
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

# 定义参数变量
options = {
  # 'detect_direction': 'true',
  'language_type': 'ENG+JPN',
}

# 调用通用文字识别接口
result = aipOcr.basicAccurate(get_file_content(filePath), options)

print(result)

with open(f'{filePath}.txt', 'wb') as f:
    f.write(str(result).encode('utf-8'))