import os
import numpy as np
import pandas as pd
import random

class_dict = {'aircraft_carrier': 0, 'airplane': 1, 'apple': 2, 'asparagus': 3, 'axe': 4, 'banana': 5, 'bat': 6, 'bear': 7, 'bed': 8, 'bee': 9, 'bicycle': 10, 'bird': 11, 'birthday_cake': 12, 'bowtie': 13, 'bridge': 14, 'broccoli': 15, 'bus': 16, 'bush': 17, 'butterfly': 18, 'calendar': 19, 'camel': 20, 'carrot': 21, 'castle': 22, 'cello': 23, 'chair': 24, 'clarinet': 25, 'clock': 26, 'cloud': 27, 'couch': 28, 'cow': 29, 'crab': 30, 'cup': 31, 'dog': 32, 'dolphin': 33, 'dresser': 34, 'elephant': 35, 'eraser': 36, 'fish': 37, 'floor_lamp': 38, 'flying_saucer': 39, 'frying_pan': 40, 'guitar': 41, 'hamburger': 42, 'hammer': 43, 'helicopter': 44, 'horse': 45, 'hot_air_balloon': 46, 'house': 47, 'ice_cream': 48, 'jacket': 49, 'kangaroo': 50, 'keyboard': 51, 'lion': 52, 'lobster': 53, 'map': 54, 'marker': 55, 'mosquito': 56, 'motorbike': 57, 'mountain': 58, 'mouse': 59, 'mushroom': 60, 'ocean': 61, 'octopus': 62, 'onion': 63, 'panda': 64, 'pants': 65, 'pear': 66, 'pencil': 67, 'pickup_truck': 68, 'pizza': 69, 'rabbit': 70, 'raccoon': 71, 'river': 72, 'sandwich': 73, 'saw': 74, 'scissors': 75, 'scorpion': 76, 'screwdriver': 77, 'shark': 78, 'shorts': 79, 'skyscraper': 80, 'snail': 81, 'sock': 82, 'spider': 83, 'squirrel': 84, 'strawberry': 85, 'table': 86, 'teapot': 87, 'telephone': 88, 'television': 89, 'tiger': 90, 'train': 91, 'trombone': 92, 'violin': 93, 'watermelon': 94, 'whale': 95, 'windmill': 96, 'wine_bottle': 97, 'wine_glass': 98, 'zebra': 99}

def create_new_test_txt():

    dst_path = '/data/input-ai/datasets/domain_net_cl/version2'
    old_path = '/data/input-ai/datasets/domain_net'
    tasks_lst = ["sketch", 'clipart', 'real', 'infograph', 'painting', 'quickdraw']

    suffix = '_test.txt'

    path = '/volumes2/continual_learning/dataset/version2/domain_test.xlsx'
    df = pd.read_excel(path)
    df = df.sort_values('classes')
    df = df.reset_index(drop=True)
    cls_lst = list(df['classes'])

    for task in tasks_lst:

        filepath = os.path.join(old_path, '{}{}'.format(task, suffix))
        file = open(filepath, 'r')
        lines = file.readlines()
        df_val = df[task]

        cls_lst.sort()
        for label_idx, cls in enumerate(cls_lst):
            class_dict[cls] = label_idx
            index = cls_lst.index(cls)
            val = df_val[index]

            new_lines = []
            for i in range(len(lines)):
                name, label = lines[i].split()
                _, name, img = name.split('/')

                if name == cls:
                    new_line = lines[i].split()
                    new_line[-1] = str(label_idx)
                    new_line = ' '.join(new_line) + '\n'

                    #print(lines[i], new_line)
                    new_lines.append(new_line)

            if val < len(new_lines):
                new_lines = random.sample(new_lines, val)

            with open(os.path.join(dst_path, '{}{}'.format(task, suffix)), 'a') as f:
                for line in new_lines:
                    f.write(line)

def create_new_train_txt():

    dst_path = '/data/input-ai/datasets/domain_net_cl/version2'

    path = '/volumes2/continual_learning/dataset/version2/domain_train.xlsx'
    df = pd.read_excel(path)
    df = df.sort_values('classes')
    df = df.reset_index(drop=True)
    path = '/data/input-ai/datasets/domain_net'
    cls_lst = list(df['classes'])

    tasks_lst = ["sketch", 'clipart', 'real', 'infograph', 'painting', 'quickdraw']

    train_suffix = '_train.txt'

    for task in tasks_lst:

        df_val = df[task]
        filepath = os.path.join(path, '{}{}'.format(task, train_suffix))
        file = open(filepath, 'r')
        lines = file.readlines()

        cls_lst.sort()
        for label_idx, cls in enumerate(cls_lst):
            class_dict[cls] = label_idx
            index = cls_lst.index(cls)
            val = df_val[index]

            new_lines = []
            for i in range(len(lines)):
                name, label = lines[i].split()
                _, name, img = name.split('/')

                if name == cls:
                    new_line = lines[i].split()
                    new_line[-1] = str(label_idx)
                    new_line = ' '.join(new_line) + '\n'

                    #print(lines[i], new_line)
                    new_lines.append(new_line)

            if val < len(new_lines):
                new_lines = random.sample(new_lines, val)

            with open(os.path.join(dst_path, '{}{}'.format(task, train_suffix)), 'a') as f:
                for line in new_lines:
                    f.write(line)

def sample_test():
    path = '/volumes2/continual_learning/dataset/domain_clean_test.xlsx'
    df = pd.read_excel(path, sheet_name='selected')
    # print(df)
    tasks = ['sketch', 'clipart', 'real', 'infograph', 'painting', 'quickdraw']

    for cat in range(20):
        sc = cat+1
        print(sc)
        group = df.loc[df['Number'] == sc]
        fin = {}

        for t in tasks:
            val_per_dom = group[t]
            cls_lst = group['classes']

            fin[t] = {}
            # fin[t]['sum'] = 0
            orig = []
            diff = []
            cnt = 0
            sum = 0

            for i, c in enumerate(cls_lst):
                ind = val_per_dom.index[i]
                orig.append(val_per_dom[ind])

                if val_per_dom[ind] >= 25:
                    fin[t][c] = 25
                    diff.append(val_per_dom[ind] - 25)
                else:
                    fin[t][c] = val_per_dom[ind]
                    diff.append(0)

                # sum += fin[t][c]
                # fin[t]['sum'] += fin[t][c]

            cnt = np.count_nonzero(diff)
            if cnt and sum < 50:
                extra = 50 - sum
                samp = int(extra / cnt)
                for i, c in enumerate(cls_lst):
                    if diff[i] > samp:
                        fin[t][c] += samp
                        # fin[t]['sum'] += samp
                    else:
                        fin[t][c] += diff[i]
                        # fin[t]['sum'] += diff[i]

        # fin_lst.append(fin)
        if cat == 0:
            df_out = pd.DataFrame(fin)
        else:
            df_out= df_out.append(pd.DataFrame(fin))
        df_out.to_excel('/volumes2/continual_learning/dataset/version2/domain_test.xlsx')

def sample():
    path = '/volumes2/continual_learning/dataset/domain_clean.xlsx'
    df = pd.read_excel(path, sheet_name='selected')
    # print(df)
    tasks = ['sketch', 'clipart', 'real', 'infograph', 'painting', 'quickdraw']

    for cat in range(20):
        sc = cat+1
        print(sc)
        group = df.loc[df['Number'] == sc]
        fin = {}

        for t in tasks:
            val_per_dom = group[t]
            cls_lst = group['classes']

            fin[t] = {}
            # fin[t]['sum'] = 0
            orig = []
            diff = []
            cnt = 0
            sum = 0

            for i, c in enumerate(cls_lst):
                ind = val_per_dom.index[i]
                orig.append(val_per_dom[ind])

                if val_per_dom[ind] >= 125:
                    fin[t][c] = 125
                    diff.append(val_per_dom[ind] - 125)
                else:
                    fin[t][c] = val_per_dom[ind]
                    diff.append(0)

                sum += fin[t][c]
                # fin[t]['sum'] += fin[t][c]

            cnt = np.count_nonzero(diff)
            if cnt and sum < 625:
                extra = 625 - sum
                samp = int(extra / cnt)
                for i, c in enumerate(cls_lst):
                    if diff[i] > samp:
                        fin[t][c] += samp
                        # fin[t]['sum'] += samp
                    else:
                        fin[t][c] += diff[i]
                        # fin[t]['sum'] += diff[i]

        # fin_lst.append(fin)
        if cat == 0:
            df_out = pd.DataFrame(fin)
        else:
            df_out= df_out.append(pd.DataFrame(fin))
        df_out.to_excel('/volumes2/continual_learning/dataset/version2/domain_train.xlsx')

def orig_test():

    path = '/data/input-ai/datasets/domain_net_cl'
    writer = pd.ExcelWriter('/volumes2/continual_learning/dataset/domain_clean_test.xlsx', engine='xlsxwriter')

    tasks_lst = ['sketch', 'clipart', 'real', 'infograph', 'painting', 'quickdraw']
    suffix = '_test.txt'

    cls_dict = {}
    for task in tasks_lst:
        filepath = os.path.join(path, '{}{}'.format(task, suffix))
        file = open(filepath, 'r')
        lines = file.readlines()

        cls_lst = []
        cls_dict[task] = {}
        cls_lst = [line.split()[0].split('/')[1] for line in lines]
        cls_uniq = sorted(set(cls_lst))
        for i in cls_uniq:
            n = cls_lst.count(i)
            cls_dict[task][i] = n

    #selected
    classes = ['mouse', 'squirrel', 'rabbit', 'dog', 'raccoon', 'tiger', 'bear', 'lion', 'panda', 'zebra', 'camel', 'horse', 'kangaroo', 'elephant', 'cow', 'whale', 'shark',
               'fish', 'dolphin', 'octopus', 'snail', 'scorpion', 'spider', 'lobster', 'crab', 'bee', 'butterfly', 'mosquito', 'bird', 'bat', 'bus', 'bicycle', 'motorbike',
               'train', 'pickup_truck', 'airplane', 'flying_saucer', 'aircraft_carrier', 'helicopter', 'hot_air_balloon', 'strawberry', 'banana', 'pear', 'apple', 'watermelon',
               'carrot', 'asparagus', 'mushroom', 'onion', 'broccoli', 'trombone', 'violin', 'cello', 'guitar', 'clarinet', 'chair', 'dresser', 'table', 'couch', 'bed',
               'clock', 'floor_lamp', 'telephone', 'television', 'keyboard', 'saw', 'axe', 'hammer', 'screwdriver', 'scissors', 'bowtie', 'pants', 'jacket', 'sock', 'shorts', 'skyscraper',
               'windmill', 'house', 'castle', 'bridge', 'cloud', 'bush', 'ocean', 'river', 'mountain', 'birthday_cake', 'hamburger', 'ice_cream', 'sandwich', 'pizza', 'calendar',
               'marker',  'map', 'eraser', 'pencil', 'wine_bottle', 'cup', 'teapot', 'frying_pan', 'wine_glass']

    cls_dict_sel = {}
    for t in tasks_lst:
        cls_dict_sel[t] = {}
        for cls in classes:
            cls_dict_sel[t][cls] = cls_dict[t][cls]

    df = pd.DataFrame.from_dict(cls_dict_sel) #, index=tasks_lst)
    df.to_excel(writer, sheet_name='selected')

    writer.save()

def orig_train():

    path = '/data/input-ai/datasets/domain_net'
    writer = pd.ExcelWriter('/volumes2/continual_learning/domain_clean.xlsx', engine='xlsxwriter')

    tasks_lst = ['sketch', 'clipart', 'real', 'infograph', 'painting', 'quickdraw']

    train_suffix = '_train.txt'
    test_suffix = '_test.txt'

    cls_dict = {}
    for task in tasks_lst:
        filepath = os.path.join(path, '{}{}'.format(task, train_suffix))
        file = open(filepath, 'r')
        lines = file.readlines()

        cls_lst = []
        cls_dict[task] = {}
        cls_lst = [line.split()[0].split('/')[1] for line in lines]
        cls_uniq = sorted(set(cls_lst))
        for i in cls_uniq:
            n = cls_lst.count(i)
            cls_dict[task][i] = n

    df1 = pd.DataFrame.from_dict(cls_dict) #, index=tasks_lst)
    df1.to_excel(writer, sheet_name='orig')

    #selected
    classes = ['mouse', 'squirrel', 'rabbit', 'dog', 'raccoon', 'tiger', 'bear', 'lion', 'panda', 'zebra', 'camel', 'horse', 'kangaroo', 'elephant', 'cow', 'whale', 'shark',
               'fish', 'dolphin', 'octopus', 'snail', 'scorpion', 'spider', 'lobster', 'crab', 'bee', 'butterfly', 'mosquito', 'bird', 'bat', 'bus', 'bicycle', 'motorbike',
               'train', 'pickup_truck', 'airplane', 'flying_saucer', 'aircraft_carrier', 'helicopter', 'hot_air_balloon', 'strawberry', 'banana', 'pear', 'apple', 'watermelon',
               'carrot', 'asparagus', 'mushroom', 'onion', 'broccoli', 'trombone', 'violin', 'cello', 'guitar', 'clarinet', 'chair', 'dresser', 'table', 'couch', 'bed',
               'clock', 'floor_lamp', 'telephone', 'television', 'keyboard', 'saw', 'axe', 'hammer', 'screwdriver', 'scissors', 'bowtie', 'pants', 'jacket', 'sock', 'shorts', 'skyscraper',
               'windmill', 'house', 'castle', 'bridge', 'cloud', 'bush', 'ocean', 'river', 'mountain', 'birthday_cake', 'hamburger', 'ice_cream', 'sandwich', 'pizza', 'calendar',
               'marker',  'map', 'eraser', 'pencil', 'wine_bottle', 'cup', 'teapot', 'frying_pan', 'wine_glass']

    cls_dict_sel = {}
    for t in tasks_lst:
        cls_dict_sel[t] = {}
        for cls in classes:
            cls_dict_sel[t][cls] = cls_dict[t][cls]

    df2 = pd.DataFrame.from_dict(cls_dict_sel) #, index=tasks_lst)
    df2.to_excel(writer, sheet_name='selected')

    writer.save()

if __name__ == '__main__':
    create_new_test_txt()