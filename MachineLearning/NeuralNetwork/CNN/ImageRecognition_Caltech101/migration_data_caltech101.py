import os, glob, shutil
import numpy as np

np.random.seed(2016)

#元データpath
r_path = './101_ObjectCategories'

#生成データpath
o_path = '../Caltech-101'

#データ読み込み
path = '%s/*/*.jpg'r_path
files = sorted(glob.glob(path))
files = np.array(files)

#使用ラベル
use_labels = ['airplanes', 'Motorbikes','Faces_easy','watch','Leopards','bonsai']
labels_count = [0,0,0,0,0,0]
#学習，評価，テストに使用する件数を指定
train_nums = [80,80,44,24,20,13]
valid_nums = [80,80,44,24,20,13]
test_nums = [640,638,348,191,160,102]

#train+validとtestにデータ分割

#データ格納ディレクトリ作成
for i in range(0,len(use_labels)):

    if not os.path.exists('%s/train_org/%i'%(o_path, i)):
        os.makedirs('%s/train_org/%i'%(o_path, i))

    if not os.path.exists('%s/test/%i'%(o_path, i)):
        os.makedirs('%s/test/%i'%(o_path, i))

#ファイル数分処理
for fl in files:

    #ファイル名取得
    filename = os.path.basename(fl)

    #親ディレクトリ ラベル取得
    parent_dir = os.path.split(os.path.split(fl)[0])[1]

    if parent_dir in use_labels:

        ind = use_labels.index(parent_dir)
        num = labels_count[ind]
        valid_num = valid_nums[ind]
        test_num = test_nums[ind]

        if num < train_nums[ind] + valid_nums[ind]:
            cp_path = '%s/train_org/%i/'%(o_path, ind)
            shutil.copy(fl, cp_path)
        else:
            cp_path = '%s/test/%i/'%(o_path, ind)
            shutil.copy(fl, cp_path)

        labels_count[ind] += 1

    else:
        #今回使用するラベル以外は無視する
        continue


#trainをtrainとvalidにデータ分割
#ho数分処理
for ho in range(0,2):

    for ii in range(0, len(use_labels)):

        #ディレクトリが存在しなかったら作成
        if not os.path.exists('%s/train/%i/%i'%(o_path, ho, ii)):
            os.makedirs('%s/train/%i/%i'%(o_path, ho, ii))
        if not os.path.exists('%s/valid/%i/%i' % (o_path, ho, ii)):
            os.makedirs('%s/valid/%i/%i' % (o_path, ho, ii))

        #データ読み込み
        path = '%s/train_org/%i/*.jpg'%(o_path, ii)
        files = sorted(glob.glob(path))
        files = np.array(files)

        perm = np.random.permutation(len(files))
        random_train = files[perm]

        train_files = random_train[:train_nums[ii]]
        valid_files = random_train[train_nums[ii]:]

        #trainデータを配置する
        for file in train_files:

            #ファイル名取得
            filename = os.path.basename(file)

            #親ディレクトリ ラベル 取得
            p_dir = os.path.split(os.path.split(file)[0])[1]

            shutil.copy(file, '%s/train/%i/%i'%(o_path, ho, int(p_dir)))

        # validデータを配置する
        for file in valid_files:
            # ファイル名取得
            filename = os.path.basename(file)

            # 親ディレクトリ ラベル 取得
            p_dir = os.path.split(os.path.split(file)[0])[1]

            shutil.copy(file, '%s/valid/%i/%i' % (o_path, ho, int(p_dir)))


#ラベルへの紐付け
str = '0:airplanes, 1:Motorbikes, 2:Faces_easy, 3:watch, 4:Leopards, 5:bonsai'
f = open('%s/label.csv'%o_path, 'w')
f.write(str)
f.close()