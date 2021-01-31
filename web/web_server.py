import os,sys
from http.server import CGIHTTPRequestHandler, HTTPServer

#设置属性
webdir="."
port=8000

#启动http 服务
os.chdir(webdir)
server_addr=("",port)
server_obj=HTTPServer(server_addr, CGIHTTPRequestHandler)
server_obj.serve_forever()