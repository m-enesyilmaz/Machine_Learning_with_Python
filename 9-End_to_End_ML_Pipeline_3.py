################################################
# End-to-End Diabetes Machine Learning Pipeline III
################################################
# her şey bitti diyelim modeli birisine gönderdik ve o da çalıştırmak istiyor
# yani canlı sistemlere entegre etmek istiyor, bunu nasıl yaparıza bakıyoruz:

import joblib
import pandas as pd

df = pd.read_csv("datasets/diabetes.csv")

random_user = df.sample(1, random_state=45)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)
# burada bir hata aldık boyut hatası bunun sebebi ise;
# modeli kaydettiğimiz yapımızda yeni feature ler üretmiştik
# burada ise dışardan bir veri alıyoruz (örnek olarak aslında ilk halinden) yani boyutlar uyuşmuyor
# çözüm olarak yeni verisetinin eski veri setiyle aynı olması lazım dönüşütürülmesi lazım

from 8-End_to_End_ML_Pipeline_2 import diabetes_data_prep

X, y = diabetes_data_prep(df)  # --> pipeline içerisinden getiriyoruz
# prediction yerine başka kaynaklarda scoring diye de geçmekte

random_user = X.sample(1, random_state=50)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)
