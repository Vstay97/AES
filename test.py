import requests
resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                     json={
                         "token": "69b10b3b219c",
                         "title": "eg. 测试title"+a,
                         "name": "eg. 测试name",
                         "content": "eg. 测试content"
                     })
print(resp.content.decode())