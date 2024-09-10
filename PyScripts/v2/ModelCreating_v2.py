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
needFitGridModel = False
needFitFinalModel = True

needSaveFinalNormaliser = True
needSaveFinalModel = True
needSaveFinalNormaliserOnnx = True
needSaveFinalModelOnnx = True
version = '2'
cleared_Path = fr'D:\Khabarov\Скрипты\6.Валидация АР\DataSets\ClearedDatasets\dataset_ar_cleared_v{version}.txt'

scaler_path = r'D:\Khabarov\Скрипты\6.Валидация АР\PyScalers'
scaler_save_Name = fr'\normalizerAR_v{version}.save'

model_path = r'D:\Khabarov\Скрипты\6.Валидация АР\PyModels'
model_joblib_name = fr'\modelAR_v{version}.joblib'

scaler_onnx_name = fr'\normalizerAR_v{version}.onnx'
model_onnx_name = fr'\modelAR_v{version}.onnx'


#region Алгоритм и гиперпараметры
#Алгоритм
log_model = LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42,n_jobs=-1)
#Гиперпараметры
penalty = ['elasticnet']
solver = ['saga']
l1_ratio_v1 = [1]
#Сетка гиперпараметров
param_grid = {'penalty': penalty,'solver':solver,'l1_ratio':l1_ratio_v1}

#endregion

#region Настройки отображения
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)
#endregion
#region Колонки
typePN = 'Тип'
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
partitionPN = 'ПерегородкаBool'
basementPN = 'ПодвалBool'
insulationPN = 'УтеплениеBool'
parapetPN = 'ПарапетBool'
concretePN = 'БетонBool'
lluPN = 'ЛЛУBool'
premisePN = 'ПомещенияBool'
plinthPN = 'ЦокольBool'
sewingPN = 'ЗашивкаBool'
platingPN = 'ОбшивкаBool'
externalPN = 'ВнешняяBool'
freePN = 'СПBool'
ceramicPN = 'КерамичBool'

groupModelNames = ['Монолитный пилон','Монолитная стена','Монолитная фундаментная плита',
                   'Деформационный шов. Каркас монолитный',
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

#Матрица уникальных значений
x_colums_withGM = [thicknessPN,
                   parkingPN,
                   plasteringPN,
                   mountedPN,
                   innerWall,
                    partitionPN,
                   basementPN,
                   insulationPN,
                   parapetPN,
                   concretePN,
                   lluPN,
                   premisePN,
                    plinthPN,
                    sewingPN,
                    platingPN,
                   externalPN,
                   freePN,
                   ceramicPN,
                   groupModelPN]
                     # ]
x_colums = [thicknessPN,
                   parkingPN,
                   plasteringPN,
                   mountedPN,
                   innerWall,
                    partitionPN,
                   basementPN,
                   insulationPN,
                   parapetPN,
                   concretePN,
                   lluPN,
                    premisePN,
                    plinthPN,
                    sewingPN,
                    platingPN,
                    externalPN,
                    freePN,
                    ceramicPN,
                   # groupModelPN]
                     ]

#endregion

#region Просмотр данных

#Загрузка данных
df = pd.read_csv(cleared_Path,sep=';')
res = df

#Длина - 170514
res = len(df)

#Баланс классов
# Группа модели
# Штукатурка черновая                      47675
# Штукатурка фасадная                      36669
# Утепление штукатурного фасада            30661
# Перегородка. Кладка                      12338
# Перегородка ГКЛ                           9969
# Внешняя стена. Кладка                     8980
# Внутренняя стена. Кладка                  6206
# Обшивка ГКЛ                               5433
# Монолитная стена                          4974
# Невалидируемое семейство АР               2693
# Утепление цоколя штукатурного              819
# Утепление цоколя навесного                 811
# Витраж                                     726
# Утепление навесного фасада                 643
# Навесной фасад                             595
# Навесной фасад. Кладка                     570
# Зашивка ГКЛ                                551
# Утепление стен подвала                     119
# Утепление помещений                         57
# Деформационный шов. Каркас монолитный       21
# Монолитная фундаментная плита                4
res = df['Группа модели'].value_counts()


#Колонки
# ['Тип', 'Неприсоединенная высота', 'Длина', 'Объем', 'Толщина', 'ADSK_Этаж',
#  'ADSK_Номер секции', 'Группа модели', 'Объект', 'ПравГМBool', 'ПаркингBool',
#  'ШтукатурныйBool', 'НавеснойBool', 'ВнутрBool', 'ПерегородкаBool', 'Отметка',
#  'ПодвалBool', 'УтеплениеBool', 'БетонBool', 'ЛЛУBool', 'ПомещенияBool',
#        'ЦокольBool', 'ЗашивкаBool', 'ОбшивкаBool', 'ВнешняяBool', 'СПBool', 'КерамичBool']
res = df.columns

#Оставляем только нужные колонки
# ['Толщина', 'ПаркингBool', 'ШтукатурныйBool', 'НавеснойBool', 'ВнутрBool',
#  'ПерегородкаBool', 'ПодвалBool', 'УтеплениеBool', 'ПарапетBool', 'БетонBool',
#  'ЛЛУBool', 'ПомещенияBool', 'ЦокольBool', 'ЗашивкаBool', 'ОбшивкаBool', 'ВнешняяBool',
#  'СПBool', 'КерамичBool', 'Группа модели']
df = df[x_colums_withGM]
res = df.columns

#endregion
#region Обучение GridSearch
if needFitGridModel:
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
    #Создание модели
    grid_model = GridSearchCV(estimator=log_model,param_grid=param_grid,cv=5,verbose=10,scoring='f1_weighted')

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
    print('===Метрики===')
    print(metricsRep)

    #Параметры модели
    print('===Параметры модели===')
    print(grid_model.best_params_)

    #Коэффициенты
    # print('===Коэффициенты===')
    # print(grid_model.best_estimator_.coef_)

    #Параметры 2
    # print('===Параметры модели get_params===')
    # print(grid_model.best_estimator_.get_params(deep=True))




    #endregion
#endregion
#region Результаты

# {'l1_ratio': 1, 'penalty': 'elasticnet', 'solver': 'saga'}
# ===Метрики===
                                       # precision    recall  f1-score   support
#                                Витраж       1.00      1.00      1.00        34
#                 Внешняя стена. Кладка       1.00      1.00      1.00       442
#              Внутренняя стена. Кладка       0.86      1.00      0.93       298
# Деформационный шов. Каркас монолитный       0.14      1.00      0.25         2
#                           Зашивка ГКЛ       1.00      1.00      1.00        27
#                      Монолитная стена       1.00      1.00      1.00       235
#         Монолитная фундаментная плита       0.50      1.00      0.67         1
#                        Навесной фасад       1.00      1.00      1.00        36
#                Навесной фасад. Кладка       1.00      1.00      1.00        30
#           Невалидируемое семейство АР       0.47      1.00      0.64       132
#                           Обшивка ГКЛ       1.00      1.00      1.00       265
#                       Перегородка ГКЛ       1.00      0.95      0.97       559
#                   Перегородка. Кладка       1.00      0.97      0.98       593
#            Утепление навесного фасада       1.00      1.00      1.00        29
#                   Утепление помещений       0.55      1.00      0.71         6
#                Утепление стен подвала       0.27      1.00      0.43         3
#            Утепление цоколя навесного       0.96      0.90      0.93        30
#         Утепление цоколя штукатурного       0.91      0.97      0.94        32
#         Утепление штукатурного фасада       0.96      0.91      0.94      1496
#                   Штукатурка фасадная       0.93      0.88      0.91      1869
#                   Штукатурка черновая       0.93      0.94      0.93      2407


#endregion



#region Обучение и сохранение финальной модели
if needFitFinalModel:
    #region Создание финальной модели
    df = df[x_colums_withGM]

    #Забираем признаки
    X = df.drop('Группа модели',axis=1)

    #Забираем целевую переменную
    y = df['Группа модели']

    #Создание scaler для масштабирования данных
    final_scaler = MinMaxScaler()

    #Учим нормализовывать значения
    final_scaler.fit(X)

    #Получаем нормализованные значения
    X_full_scaled =  final_scaler.transform(X)

    #Создание модели
    final_model = LogisticRegression(max_iter=2000
                                     ,class_weight='balanced'
                                     ,random_state=42
                                     ,n_jobs=-1
                                     ,l1_ratio=1
                                     ,penalty='elasticnet'
                                     ,solver='saga')
    #Обучение
    final_model.fit(X_full_scaled,y)
    print('Финальная модель обучена')



    #endregion
    #region Сохранение модели и нормализатора joblib

    #Сохранение нормализатора
    if needSaveFinalNormaliser:
        dump(final_scaler, scaler_path + scaler_save_Name)
        print('Нормализатор joblib сохранен')


    #Сохранение модели
    if needSaveFinalModel:
        dump(final_model, model_path + model_joblib_name)
        print('Модель joblib сохранена')

    #endregion
    #region Сохранение модели и нормализатора onnx

    #region Нормализатор
    if needSaveFinalNormaliserOnnx:
        ONNXNormalizerPath = scaler_path + scaler_onnx_name
        initial_type = [('feature_input',FloatTensorType([None,18]))]
        onnxNormalizer = convert_sklearn(final_scaler,initial_types=initial_type)
        with open(ONNXNormalizerPath,"wb") as f:
            f.write(onnxNormalizer.SerializeToString())
        print('Нормализатор onnx сохранен')
    #endregion
    #region Модель

    if needSaveFinalModelOnnx:
        ONNXModelPath = model_path + model_onnx_name
        initial_type = [('feature_input', FloatTensorType([None, 18]))]
        onnx = convert_sklearn(final_model, initial_types=initial_type)

        with open(ONNXModelPath, "wb") as f:
            f.write(onnx.SerializeToString())
        print('Модель onnx сохранена')

    #endregion

    #endregion
#endregion


print(res)