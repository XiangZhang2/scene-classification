import json
import random
from utils import get_json


def label_shuffle(annotation_list):
    
    image_list_dic = {}
    image_num = []
    classes = 80

    for i in range(classes):
        image_list_dic[str(i)] = []
        for j in range(len(annotation_list)):
            if annotation_list[j][0] == i:
                image_list_dic[str(i)].append(annotation_list[j][1])
        image_num.append(len(image_list_dic[str(i)]))

    max_index = image_num.index(max(image_num))  #label_id=32  image_num=862
    min_index = image_num.index(min(image_num))  #label_id=55  image_num=168
    num_max = image_num[max_index]

    for n in range(classes):
        multiple = num_max // image_num[n]
        mod = num_max % image_num[n]
        copy = image_list_dic[str(n)]
        
        if multiple > 1:
            for m in range(multiple-1):
                image_list_dic[str(n)] = image_list_dic[str(n)] + copy
        
        image_list_dic[str(n)] += random.sample(copy, mod)

    annotation_list_with_shuffle = []
    for k in range(classes):
        for z in range(len(image_list_dic[str(k)])):
            annotation = [int(k), image_list_dic[str(k)][z]]
            annotation_list_with_shuffle.append(annotation)
    #print len(annotation_list_with_shuffle) # 689690

    return annotation_list_with_shuffle


























