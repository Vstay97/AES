ip = {}


for i in f.readlines():
    ip_attr = i.strip().split()[0]
    if ip_attr in ip.keys():  # 如果ip存在于字典中，则将该ip的value也就是个数进行增加
        ip[ip_attr] = ip[ip_attr] + 1
    else:
        ip[ip_attr] = 1
s = sorted(ip.items(), key=lambda x: x[1], reverse=True)
print(s)

# for  value  in  sorted(ip.values()):
#     for key in ip.keys():
#         if ip[key]==value:
#             print(key,ip[key])
print(ip)