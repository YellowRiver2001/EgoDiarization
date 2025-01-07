



#生成短期追踪id和长期追踪的对应字典
filename = 'short2global.txt'
id_dict = {}
with open(filename, 'r') as file:
    for line in file:
        key, value = line.strip().split(':')
        id_dict[int(key)] = int(value)

id_set=set([])
for id in id_dict.values():
    id_set.add(id)

sorted_list = sorted(list(id_set))

id_index={}
for i in range(len(sorted_list)):
    id_index[sorted_list[i]] = i

for key,value in id_dict.items():
    id_dict[key] = id_index[value]

print(id_dict)
