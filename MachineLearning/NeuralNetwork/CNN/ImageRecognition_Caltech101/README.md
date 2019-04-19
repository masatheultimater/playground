# CNN for Image Recognition

## Dataset  

### data
- 101_ObjectCategories(RAW data)

### description
- Image dataset from Caltech
- 8,677 pics(300*200px RGB)
- 101 categories
- 130MB

### attention
- extract 6 categories has many Images only

### extracted dataset  

|No|category|pics|
|---|---|---|
|0|airplanes|800|
|1|Motorbikes|798|
|2|Faces_easy|435|
|3|watch|239|
|4|Leopards|200|
|5|bosai|128|
|Total|-|2,600|

|class|train|valid|test|total|
|---|---|---|---|---|
|airplanes|80|80|640|800|
|Motorbikes|80|80|638|798|
|Faces_easy|44|43|348|435|
|watch|24|24|191|239|
|Leopards|20|20|160|200|
|bonsai|13|13|102|128|
|total|261|260|2,079|2,600|

## model
- using 3 models to comparison
  - scratch made CNN
  - trained CNN
    - VGG-16
    - ResNet-152

#### programs

|function|data|language|FW|program|description|
|---|---|---|---|---|---|
|data making|caltech101|Python|-|migration_data_caltech101.py|extract 6 classes|
|data making|caltech101|Python|-|data_augmentation.py|data augmentation|
|classify the images|6 classed data|Python|Keras|9_Layer_CNN.py|9 layered NN|
|classify the images|6 classed data|Python|Keras|VGG_16.py|VGG_16|
|classify the images|6 classed data|Lua|Torch|main.lua|ResNet-152|
|classify the images|6 classed data|Lua|Torch|opts.lua|ResNet-152|
|classify the images|6 classed data|Lua|Torch|dataloader.lua|ResNet-152|
|classify the images|6 classed data|Lua|Torch|datasets/caltech101-gen.lua|ResNet-152|
|classify the images|6 classed data|Lua|Torch|datasets/caltech101.lua|ResNet-152|
|classify the images|6 classed data|Lua|Torch|models/init.lua|ResNet-152|
|classify the images|6 classed data|Lua|Torch|average_outputs.py|ResNet-152|
|classify the images|6 classed data|Lua|Torch|(and more)|ResNet-152|
|classify the images|6 classed data|Python|Keras|multiple_model.py|average results|
|classify the images|6 classed data|Python|Keras|average_3models.py|average results|
|classify the images|6 classed data|Python|Keras|make_pseudo_label.py|make pseudo label to upgrade accuracy(Stacked Generalization)|
|classify the images|6 classed data|Python|Keras|pseudo_model.py|make pseudo label|

### scratch made CNN

#### framework
- Keras

#### model structure
- 9 layer

### validation method
- 2 hold-outs
  - 1st hold-out : training data,validation data 
  - 2nd hold-out : re-choose in same ratio from shuffled data

### how to handle the data
- extract 6 categories
- augmentate data to 5 times
  - 1,305 training data (/ho)
  - 10,395 test data (/ho)
- clasify for 20,790 pics, 2,079 results(arithmetic average)


## Steps for experiment
### environment
1. Local notebook
  - Linux(virtualbox) on windows PC
  - CPU only
2. Google Colaboratory
  - Ubuntu 18.04
  - GPU mode

### record
- jupyter notebook

### Libraries
- Theano
- Keras
- Open-CV
- scikit-image
- Numpy

1. Data handling  
**migration_data_caltech101.py**  
- Extract 6 catogories, 2,600 pics from RAW data to directories above
- destribute data to 3 detasets randomly
  - training data(261 pics)
  - validation data(260 pics)
  - test data(2079 pics)
    - using few data cause huge training data may cause almost same accuracy rate in this high qualified dataset
- can modify the numbers in each datasets and categories(train,valid,test)
```migration_data_caltech101.py
train_nums = [80,80,44,24,20,13]
valid_nums = [80,80,43,24,20,13]
test_nums = [640,638,348,191,160,102]
```
- directory for RAW data
  - /home/user/data/101_ObjectCategories
- directory for extracted data
  - /home/user/data/Caltech-101
    - label.csv
    - test(test data)
      - 0(airplanes class)
      - 1
      - 2
      - 3
      - 4
      - 5
    - train(training data)
      - 0(hold-out 1)
      - 1
    - train_org(copied 6 categories)
      - 0(airplanes class)
      - 1
      - 2
      - 3
      - 4
      - 5      
    - valid(validation data)
      - 0(hold-out 1)
      - 1

- directory for augmented data
  - /home/user/data/Caltech-101
    - label.csv
    - test(test data)
      - 0(airplanes class)
      - 1
      - 2
      - 3
      - 4
      - 5
    - train(training data)
      - 0(hold-out 1)
      - 1
    - train_org(copied 6 categories)
      - 0(airplanes class)
      - 1
      - 2
      - 3
      - 4
      - 5      
    - valid(validation data)
      - 0(hold-out 1)
      - 1



|directory|categories|
|0|airplanes|
|1|Motorbikes|
|2|Faces_easy|
|3|watch|
|4|Leopards|
|5|bonsai|





2. 
3. 
4. 
5. 

