import json

# 读取 JSON 文件
with open('extraction_alpha.json', 'r') as file:
    data = json.load(file)

#print(0,len(data['intents']))
for i in range(0,len(data['intents'])):
    item=data['intents'][i]
    for j in range(0,len(item)):
        detail=item['parts'][j]
        print('111')
