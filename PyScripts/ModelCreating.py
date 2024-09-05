import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump,load
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt


res = ''
needSaveFinalNormaliser = False
needSaveFinalModel = False
needSaveFinalNormaliserOnnx = False
needSaveFinalModelOnnx = False
version = '1'


#region Настройки отображения
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)
#endregion
#region Колонки
groupModelPN = 'Группа модели'
heightPN = 'Неприсоединенная высота'
lengthPN = 'Длина'
volumePN = 'Объем'
thicknessPN = 'Толщина'
floorPN = 'ADSK_Этаж'
sectionPN = 'ADSK_Номер секции'

parkingPN = 'ПаркингBool'
plasteringPN = 'ШтукатурныйBool'
mountedPN = 'НавеснойBool'
gmCorrect = 'ПравГМBool'
innerWall = 'ВнутрBool'
basementPN = 'ПодвалBool'
insulationPN = 'УтеплениеBool'
parapetPN = 'ПарапетBool'
concretePN = 'БетонBool'
lluPN = 'ЛЛУBool'
premisePN = 'ПомещенияBool'

groupModelNames = ['Монолитный пилон','Монолитная стена',
                   'Внешняя стена. Кладка','Внутренняя стена. Кладка',
                   'Перегородка. Кладка','Штукатурка фасадная',
                   'Навесной фасад','Навесной фасад. Кладка',
                   'Архитектурные элементы. Специальные элементы','Отлив',
                   'Витраж','Перегородка ГКЛ',
                   'Обшивка ГКЛ','Зашивка ГКЛ',
                   'Штукатурка черновая','Утепление штукатурного фасада',
                   'Утепление навесного фасада','Утепление цоколя штукатурного',
                   'Утепление цоколя навесного','Утепление помещений',
                   'Невалидируемое семейство АР',
                   'Утепление стен подвала']
gmErrorMap = {'BRU_ФасадШтукатурный_Штукатурка_10мм':'Штукатурка фасадная',
              'Монолитная стена подвала':'Монолитная стена',
              'Монолитная стена лестничного-лифтового узла':'Монолитная стена',
              'Монолитная стена приямка':'Монолитная фундаментная плита',
              'Элемент фасада':'Архитектурные элементы. Специальные элементы',
              'Отделка стен':'Штукатурка черновая'}

x_colums = [thicknessPN,
                   parkingPN,
                   plasteringPN,
                   mountedPN,
                   innerWall,
                   basementPN,
                   insulationPN,
                   parapetPN,
                   concretePN,
                   lluPN,
                    premisePN,
                   # groupModelPN]
                     ]
#endregion


#region Загрузка и проверка данных
filePath = r'D:\Khabarov\Скрипты\6.Валидация АР\DataSets\ClearedDatasets'
fileName = r'\dataset_ar_cleared_v1.txt'
df = pd.read_csv(filePath+fileName,sep=';')

#Просмотр данных
#            Тип  Неприсоединенная высота  ...  ЛЛУBool  ПомещенияBool
# 0  BRU_ЛЛУ_160                   2350.0  ...     True          False
# 1  BRU_ЛЛУ_160                   2350.0  ...     True          False
res = df.head(5)

#len - 117753
res = len(df)


#Баланс классов
# Группа модели
# Штукатурка черновая              32655
# Штукатурка фасадная              29326
# Утепление штукатурного фасада    21695
# Перегородка. Кладка               7387
# Перегородка ГКЛ                   6627
# Внешняя стена. Кладка             5810
# Монолитная стена                  4164
# Обшивка ГКЛ                       4155
# Внутренняя стена. Кладка          4091
# Витраж                             475
# Утепление навесного фасада         468
# Навесной фасад. Кладка             454
# Навесной фасад                     181
# Утепление стен подвала             179
# Невалидируемое семейство АР         34
# Монолитная фундаментная плита       28
# Утепление помещений                 24
res = df['Группа модели'].value_counts()

#Колонки
# ['Тип', 'Неприсоединенная высота', 'Длина', 'Объем', 'Толщина',
#        'ADSK_Этаж', 'ADSK_Номер секции', 'Группа модели', 'ПравГМBool',
#        'ПаркингBool', 'ШтукатурныйBool', 'НавеснойBool', 'ВнутрBool',
#        'Отметка', 'ПодвалBool', 'УтеплениеBool', 'ПарапетBool', 'БетонBool',
#        'ЛЛУBool', 'ПомещенияBool']
res = df.columns


#Убираем лишние колонки
df = df.drop(labels=['Тип','Неприсоединенная высота','Длина',
                     'Объем','ADSK_Номер секции','ADSK_Этаж',
                     'ПравГМBool','Отметка'],axis=1)

# ['Толщина', 'Группа модели', 'ПаркингBool', 'ШтукатурныйBool',
# 'НавеснойBool', 'ВнутрBool', 'ПодвалBool', 'УтеплениеBool',
# 'ПарапетBool', 'БетонBool', 'ЛЛУBool', 'ПомещенияBool']
res = df.columns

#endregion
#region Разбивка и Стандартизация данных

#Забираем признаки
X = df.drop('Группа модели',axis=1)

#Забираем целевую переменную
y = df['Группа модели']

#Разбивка на Валидационные + тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

#Создание scaler для масштабирования данных
scaler = MinMaxScaler()

#Учим нормализовывать значения
scaler.fit(X_train)

#Получаем нормализованные значения
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




#endregion
#region Создание модели

#Создание эстимейтора
log_model = LogisticRegression(solver='lbfgs',max_iter=5000,class_weight='balanced')

#Гиперпараметры
penalty = ['l2',None]
# C = np.logspace(0,10,10)

#Сетка гиперпараметров
param_grid = {'penalty': penalty}

#Создание модели
grid_model = GridSearchCV(estimator=log_model,param_grid=param_grid,cv=10,verbose=2)

#Обучение
grid_model.fit(X_train,y_train)

#endregion
#region Метрики

#Выдаем определение классов для тестовых данных
y_pred = grid_model.predict(X_test)

#Accuracy
acc = accuracy_score(y_test,y_pred)

#Матрица ошибок
confM = confusion_matrix(y_test,y_pred)

#Отчет по метрикам
metricsRep = classification_report(y_test,y_pred)


#endregion
#region Сохранение модели и нормализатора joblib

#Сохранение нормализатора
filePath = r'D:\Khabarov\Скрипты\6.Валидация АР\PyScalers'
fileName = fr'\normalizerAR_v{version}.save'
if needSaveFinalNormaliser:
    dump(scaler, filePath + fileName)
    print('Нормализатор joblib сохранен')


#Сохранение модели
filePath = r'D:\Khabarov\Скрипты\6.Валидация АР\PyModels'
fileName = fr'\modelAR_v{version}.joblib'
if needSaveFinalModel:
    dump(grid_model.best_estimator_, filePath + fileName)
    print('Модель joblib сохранена')

#endregion
#region Сохранение модели и нормализатора onnx

#region Нормализатор
if needSaveFinalNormaliserOnnx:
    filePath = r'D:\Khabarov\Скрипты\6.Валидация АР\PyScalers'
    fileName = fr'\normalizerAR_v{version}.onnx'
    ONNXNormalizerPath = filePath + fileName
    initial_type = [('feature_input',FloatTensorType([None,11]))]
    onnxNormalizer = convert_sklearn(scaler,initial_types=initial_type)
    with open(ONNXNormalizerPath,"wb") as f:
        f.write(onnxNormalizer.SerializeToString())
    print('Нормализатор onnx сохранен')
#endregion
#region Модель

if needSaveFinalModelOnnx:
    fileName = fr'\modelAR_v{version}.onnx'
    filePath = r'D:\Khabarov\Скрипты\6.Валидация АР\PyModels'
    ONNXModelPath = filePath + fileName

    initial_type = [('feature_input', FloatTensorType([None, 11]))]
    onnx = convert_sklearn(grid_model, initial_types=initial_type)

    with open(ONNXModelPath, "wb") as f:
        f.write(onnx.SerializeToString())
    print('Модель onnx сохранена')

#endregion

#endregion



print(res)
plt.show()