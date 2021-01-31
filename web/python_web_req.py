#通过脚本，发送http请求
from urllib.request import urlopen

#调用方法： 发送http请求
conn=urlopen("http://localhost:8000/cgi-bin/form.py?user=abc123")
res_data=conn.read()
print(res_data)
