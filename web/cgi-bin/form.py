import cgi

# 字段解析
form_data = cgi.FieldStorage()
if form_data:
    user_name = cgi.escape(form_data['user'].value)
print("Content-type:text/html\n")
print("<title>reply page</title>")

# 判断输入的数据
if not 'user' in form_data:
    print("<h1> who you are ? </h1> ")
else:
    print("<h1> hello %s </h1>" % user_name)