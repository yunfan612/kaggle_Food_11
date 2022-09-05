環境:
tensorflow-gpu 2.4.0
Keras          2.4.3

描述: 對kaggle Food 11資料庫做transfer learning (InceptionResNetV2)訓練與測試，共有11類食物


dataset下載位置:https://www.kaggle.com/datasets/trolukovich/food11-image-dataset
-------------train--------------
Bread: 994
Dairy product: 429
Dessert: 1500
Egg: 986
Fried food: 848
Meat: 1325
Noodles-Pasta: 440
Rice: 280
Seafood: 855
Soup: 1500
Vegetable-Fruit: 709

-------------validation--------------
Bread: 362
Dairy product: 144
Dessert: 500
Egg: 327
Fried food: 326
Meat: 449
Noodles-Pasta: 147
Rice: 96
Seafood: 347
Soup: 500
Vegetable-Fruit: 232

--------------evaluation-------------
Bread: 368
Dairy product: 148
Dessert: 500
Egg: 335
Fried food: 287
Meat: 432
Noodles-Pasta: 147
Rice: 96
Seafood: 303
Soup: 500
Vegetable-Fruit: 231

Transfer_InceptionResNetV2_v2.py
訓練model

Transfer_InceptionResNetV2_v2_test.py
測試model