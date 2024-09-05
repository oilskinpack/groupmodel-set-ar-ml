import re
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
from sympy.physics.units import farad
from win32con import DFCS_HOT

from PyScripts.Helpers.DfHelper import DfHelper

res = ''
needSaveFinalDf = False
version = '2'
dirPath = r'D:\Khabarov\Скрипты\6.Валидация АР\DataSets\DatasetsOfRevitModels'


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
basementPN = 'ПодвалBool'
insulationPN = 'УтеплениеBool'
parapetPN = 'ПарапетBool'
concretePN = 'БетонBool'
lluPN = 'ЛЛУBool'
premisePN = 'ПомещенияBool'

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
                   basementPN,
                   insulationPN,
                   parapetPN,
                   concretePN,
                   lluPN,
                   premisePN,
                   groupModelPN]
                     # ]
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
#region Функции

def show_not_correct_gm_values(fullDf):
    df = fullDf
    df[gmCorrect] = np.where(df[groupModelPN].isin(groupModelNames), True, False)
    gmNotMapped = df[df[gmCorrect] == False][[typePN, 'Объект', groupModelPN]].value_counts()
    res = gmNotMapped
    return res

def map_gm_values(fullDf):
    df = fullDf
    df[groupModelPN] = np.where(df[gmCorrect] == True, df[groupModelPN],
                                    df[groupModelPN].map(gmErrorMap))
    return df

#
#region Анализ данных

#region Анализ и актуализация ГруппыМодели (классы)
fullDf = DfHelper.union_all_dfs(dirPath)
# res = fullDf

#Общая информация
# res = fullDf.info()

#Пустая Группа модели
res = fullDf[fullDf[groupModelPN].isna()]

#Неактуальные группы модели
res = show_not_correct_gm_values(fullDf)

#region Заменяем неактуальные группы модели на актуальные
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('Витраж',False) == True,'Витраж',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('Зашивка',False) == True,'Зашивка ГКЛ',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('Обшивка',False) == True,'Обшивка ГКЛ',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_ФасадШтукатурный_Утеплитель',False) == True
                                ,'Утепление штукатурного фасада'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_ОтделкаПомещений',False) == True,'Штукатурка черновая'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('Монолитная стена приямка',False) == True
                                ,'Монолитная фундаментная плита'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('Монолитная стена лестнично',False) == True,'Монолитная стена'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('Монолитная стена подв',False) == True,'Монолитная стена'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_ФасадШтукатурный',False) == True
                                ,'Утепление штукатурного фасада'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_ВнешняяСтена_БлокКерамический',False) == True,'Внешняя стена. Кладка'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_Перегородка_ГКЛ',False) == True,'Перегородка ГКЛ'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_ВнешняяСтена_Кирпич',False) == True,'Внешняя стена. Кладка'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('Отделка стен',False) == True,'Штукатурка черновая'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_Перегородка_Блок',False) == True,'Перегородка. Кладка'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_ФасадЦоколь_Утеплитель',False) == True
                                ,'Утепление цоколя навесного'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_ФасадНавесной_Аквапанель',False) == True,'Навесной фасад'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_Откос_Штукатур',False) == True,'Штукатурка фасадная'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_ФасадНавесной_Утеплитель',False) == True
                                ,'Утепление навесного фасада'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('Элемент фасада',False) == True,'Навесной фасад'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_Парапет_БлокКерамический',False) == True
                                ,'Внешняя стена. Кладка'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_Парапет_Утеплитель',False) == True
                                ,'Утепление штукатурного фасада'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_ВнешняяСтена_Тех.Этаж_Утеплитель_100мм',False) == True
                                ,'Утепление стен подвала'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('Невалидируемые семейства',False) == True
                                ,'Утепление стен подвала'
                                ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[groupModelPN].str.contains('BRU_ДеформационныйШов_Утеплитель_50мм',False) == True
                                ,'Деформационный шов. Каркас монолитный'
                                ,fullDf[groupModelPN])
res = show_not_correct_gm_values(fullDf)

#Нулевая группа модели - 0
res = fullDf[fullDf[groupModelPN].isna()]
#endregion
#region Заполнение NA значений

#Процент NA значений
res = DfHelper.percent_missing(fullDf)

#Неприсоединенная высота
res = fullDf[fullDf[heightPN].isna()]
fullDf = fullDf.dropna(subset=[heightPN])
#Толщина
res = fullDf[fullDf[thicknessPN].isna()]
fullDf[thicknessPN] = fullDf[thicknessPN].fillna(0)
#Объем
fullDf[volumePN] = fullDf[volumePN].fillna(0)
#Секция
fullDf[sectionPN] = fullDf[sectionPN].fillna('Секция 1')
#Этаж
res = fullDf[fullDf[floorPN].isna()][[typePN,groupModelPN]].value_counts()
fullDf[floorPN] = fullDf[floorPN].fillna('Этаж 03 (отм. +7,200)')

#Финальная проверка
res = DfHelper.percent_missing(fullDf)


#endregion
#region Конвертация по типу

# res = fullDf.convert_dtypes().info()
fullDf = fullDf.apply(DfHelper.convertToDouble)
fullDf = fullDf.convert_dtypes()
res = fullDf.info()

#endregion
#region Конструирование признаков

#region Добавляем условие Паркинг(bool)
fullDf[parkingPN] = np.where(fullDf[sectionPN].str.contains('Секц') == True,0,1)
res = fullDf[fullDf[parkingPN] == 1][groupModelPN].value_counts()
#endregion
#region Добавляем условие Штукатурный
fullDf[plasteringPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'штук')

#Меняем группу модели
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ФасадШтукатурный_Утеплитель_150мм'
                                              ,'Утепление штукатурного фасада'
                                              ,fullDf[groupModelPN])
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ФасадШтукатурный_Утеплитель_150мм_Цоколь'
                                              ,'Утепление цоколя штукатурного'
                                              ,fullDf[groupModelPN])

res = fullDf[fullDf[plasteringPN] == 1][[groupModelPN]].value_counts()
#endregion
#region Добавляем условие Навесной



#endregion

#endregion

#endregion

#endregion



print(res)