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
needSaveUniqueValues = False
version = '2'
dirPath = r'D:\Khabarov\Скрипты\6.Валидация АР\DataSets\DatasetsOfRevitModels'
uniqTypesPath = fr'D:\Khabarov\Скрипты\6.Валидация АР\DataSets\UniqueTypes\uniqueTypes_v{version}.xlsx'
cleared_Path = fr'D:\Khabarov\Скрипты\6.Валидация АР\DataSets\ClearedDatasets\dataset_ar_cleared_v{version}.txt'


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

def getElev(value):
    res = 0
    value = str(value)
    match = re.search(r'([-+]?\d+,\d+)', value)
    if match:
        res = match.group(1).replace('+','').replace(',','.')
        res = float(res)
    return res
#endregion
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
# res = fullDf.info()

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
fullDf[mountedPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'навесн')
res = fullDf[fullDf[mountedPN] == 1][[groupModelPN]].value_counts()

#Заменяем неправильные значения Группы модели
res = DfHelper.show_unique_by_two_conditions(fullDf,mountedPN,1
                                             ,groupModelPN,'Утепление штукатурного фасада'
                                             ,[typePN,groupModelPN])
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ФасадНавесной_ШтукатуркаБелая_10мм'
                                              ,'Штукатурка фасадная'
                                              ,fullDf[groupModelPN])
fullDf[typePN] = DfHelper.replace_value(fullDf,typePN,'BRU_ФасадНавесной_Утеплитель_100мм'
                                              ,'BRU_ФасадШтукатурный_Утеплитель_100мм'
                                              ,fullDf[typePN])

# fullDf[mountedPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'навесн')
# fullDf[plasteringPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'штук')
res = fullDf[fullDf[mountedPN] == 1][[groupModelPN]].value_counts()

fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('BRU_Фасад_Утеплитель_300мм') == True
                                ,'Утепление навесного фасада',fullDf[groupModelPN])

fullDf[mountedPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'навесн')

#endregion
#region Добавляем условие Внутренняя
fullDf[innerWall] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'внутр')

#Заменяем неправильные значения Группы модели
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ВнутреняяСтена_Тех.Этаж_Утеплитель_100мм'
                                              ,'Утепление стен подвала'
                                              ,fullDf[groupModelPN])
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ВнутренняяСтена_Тамбур_Утеплитель_150мм'
                                              ,'Утепление штукатурного фасада'
                                              ,fullDf[groupModelPN])
res = fullDf[fullDf[innerWall] == 1][[typePN,groupModelPN]].value_counts()

#endregion
#region Добавляем условие Перегородка
fullDf[partitionPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'перегор')
res = fullDf[fullDf[partitionPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

#Заменяем неправильные значения Группы модели
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_Перегородка_БлокКерамический_250мм'
                                              ,'Внутренняя стена. Кладка'
                                              ,fullDf[groupModelPN])
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_Перегородка_ГКЛ_Gyproc_С-2М-2ОПТИМА_280/2х75_ГСП-А_12,5'
                                              ,'Перегородка ГКЛ'
                                              ,fullDf[groupModelPN])
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_Перегородка_ЦПВ_КНАУФ_С381_100/75_Аквапанель_12,5'
                                              ,'Перегородка ГКЛ'
                                              ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('Перегородка_ГКЛ',False) == True,'Перегородка ГКЛ',fullDf[groupModelPN])
res = fullDf[fullDf[partitionPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

#endregion
#region Добавляем условие Отметка

#Чистим значения - Утепление помещений только выше 0
fullDf['Отметка'] = fullDf[floorPN].apply(getElev)
fullDf[floorPN] = np.where( (fullDf[groupModelPN] == 'Утепление помещений') & (fullDf['Отметка'] < 0),
                            'Этаж 01 (отм. +0,000)',
                            fullDf[floorPN])
fullDf[floorPN] = np.where( (fullDf[groupModelPN] == 'Утепление стен подвала') & (fullDf['Отметка'] > 0),
                            'Этаж -01 (отм. -3,000)',
                            fullDf[floorPN])
fullDf['Отметка'] = fullDf[floorPN].apply(getElev)

#endregion
#region Добавляем условие подвал
fullDf[basementPN] = np.where(fullDf['Отметка'] >= 0,False,True)

res = fullDf[fullDf[basementPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

#endregion
#region Добавляем условие Утеплитель
fullDf[insulationPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'утепл')
res = fullDf[fullDf[insulationPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

#endregion
#region Добавляем параметр Парапет
fullDf[parapetPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'парапет')
res = fullDf[fullDf[parapetPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()
#endregion
#region Добавляем условие Бетон
fullDf[concretePN] = np.where((fullDf[typePN].str.contains('бетон', case=False) == True)
                              | (fullDf[typePN].str.contains('жб', case=False) == True)
                              , 1, 0)
res = fullDf[fullDf[concretePN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ОтделкаПомещений_ПоБетону_20мм_СухаяЗона'
                                              ,'Штукатурка черновая'
                                              ,fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('BRU_ЖБ стена_200') == True
                                ,'Монолитная стена',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('BRU_ЖБ стена_250') == True
                                ,'Монолитная стена',fullDf[groupModelPN])


res = fullDf[fullDf[concretePN] == 1][[typePN,thicknessPN,concretePN,groupModelPN]].value_counts()

#endregion
#region Добавляем условие ЛЛУ
fullDf[lluPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'ллу')
res = fullDf[fullDf[lluPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

#endregion
#region Добавляем условие Помещение
fullDf[premisePN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'помещ')
res = fullDf[fullDf[premisePN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ОтделкаПомещений_ПоУтеплителю_15мм_СухаяЗона'
                                              ,'Штукатурка черновая'
                                              ,fullDf[groupModelPN])
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ОтделкаПомещений_ПоБетону_20мм_МокраяЗона'
                                              ,'Штукатурка черновая'
                                              ,fullDf[groupModelPN])
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ОтделкаПомещений_Утеплитель_150мм'
                                              ,'Утепление цоколя навесного'
                                              ,fullDf[groupModelPN])
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ОтделкаПомещений_Утеплитель_100мм'
                                              ,'Утепление цоколя навесного'
                                              ,fullDf[groupModelPN])
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ОтделкаПомещений_Утеплитель_50мм'
                                              ,'Утепление помещений'
                                              ,fullDf[groupModelPN])
res = fullDf[fullDf[premisePN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

#endregion
#region Добавляем условие Цоколь
fullDf[plinthPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'цокол')
res = fullDf[fullDf[plinthPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ФасадЦоколь_Утеплитель_150мм'
                                              ,'Утепление цоколя навесного'
                                              ,fullDf[groupModelPN])
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ФасадЦоколь_Утеплитель_50мм'
                                              ,'Утепление цоколя навесного'
                                              ,fullDf[groupModelPN])
fullDf[groupModelPN] = DfHelper.replace_value(fullDf,typePN,'BRU_ФасадЦоколь_Утеплитель_100мм'
                                              ,'Утепление цоколя штукатурного'
                                              ,fullDf[groupModelPN])
res = fullDf[fullDf[plinthPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

#endregion
#region Добавляем условие Зашивка
fullDf[sewingPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'зашивк')
res = fullDf[fullDf[sewingPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('Зашивка_ГКЛ') == True,'Зашивка ГКЛ',fullDf[groupModelPN])
res = fullDf[fullDf[sewingPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('BRU_Зашивка_ЦПВ_КНАУФ_С685_50_Аквапанель_12,5') == True
                                ,'Зашивка ГКЛ',fullDf[groupModelPN])

#endregion
#region Добавляем условие Обшивка
fullDf[platingPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'обшивк')
res = fullDf[fullDf[platingPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('Обшивка_ГКЛ') == True,'Обшивка ГКЛ',fullDf[groupModelPN])
res = fullDf[fullDf[platingPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

#endregion
#region Добавляем условие Внешняя
fullDf[externalPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'внешн')
res = fullDf[fullDf[externalPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('BRU_ВнешняяСтена_КирпичКерамическийРядовойПолнотелый1,4НФ_120мм') == True
                                ,'Навесной фасад. Кладка',fullDf[groupModelPN])
res = fullDf[fullDf[externalPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

#endregion
#region Добавляем условие СП
fullDf[freePN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'свободн')
res = fullDf[fullDf[freePN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('Зашивка в свободной планировке_75мм') == True
                                ,'Невалидируемое семейство АР',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('Перегородка в свободной планировке_125мм') == True
                                ,'Невалидируемое семейство АР',fullDf[groupModelPN])
res = fullDf[fullDf[freePN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

#endregion
#region Добавляем условие Керамический
fullDf[ceramicPN] = DfHelper.create_bool_feature_by_contains(fullDf,typePN,'керамич')
res = fullDf[fullDf[ceramicPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('BRU_Вентканал_КирпичКерамическийРядовойПолнотелый1,4НФ_120мм') == True
                                ,'Перегородка. Кладка',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('BRU_Парапет_КирпичКерамическийРядовойПолнотелый1,4НФ_120мм') == True
                                ,'Перегородка. Кладка',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('BRU_ВнешняяСтена_КирпичКерамическийРядовойПолнотелый1,4НФ_120мм') == True
                                ,'Перегородка. Кладка',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('BRU_Парапет_БлокКерамический6,74НФ_120мм') == True
                                ,'Перегородка. Кладка',fullDf[groupModelPN])

res = fullDf[fullDf[ceramicPN] == 1][[typePN,thicknessPN,groupModelPN]].value_counts()

#endregion
#region Обработка деформационного шва
fullDf[groupModelPN] = np.where(fullDf[typePN].str.contains('дефор',False) == True
                                ,'Деформационный шов. Каркас монолитный',fullDf[groupModelPN])

fullDf = fullDf[fullDf[groupModelPN] != 'Деформационный шов. Каркас монолитный']

#endregion
#region Обработка утепления для цоколя, помещений и подвала
badIsulationTypes = fullDf[fullDf[groupModelPN].isin(['Утепление цоколя штукатурного',
                   'Утепление цоколя навесного','Утепление помещений',
                   'Утепление стен подвала'])]
res = badIsulationTypes[[typePN,thicknessPN,groupModelPN]].value_counts()

#Цоколь штукатурный
fullDf[groupModelPN] = np.where((fullDf[plinthPN] == 1) & (fullDf[plasteringPN] == 1)
                                ,'Утепление цоколя штукатурного',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where((fullDf[plinthPN] == 1) & (fullDf[plasteringPN] == 0)
                                ,'Утепление цоколя навесного',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where((fullDf[premisePN] == 1)
                                & (fullDf[insulationPN] == 1)
                                & (fullDf[thicknessPN] == 50)
                                & (fullDf[basementPN] == 0)
                                ,'Утепление помещений',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where((fullDf[premisePN] == 1)
                                & (fullDf[insulationPN] == 1)
                                & (fullDf[thicknessPN] >= 50)
                                & (fullDf[basementPN] == 1)
                                ,'Утепление стен подвала',fullDf[groupModelPN])

fullDf[groupModelPN] = np.where(fullDf[typePN] == 'BRU_ОтделкаПомещений_Утеплитель_150мм'
                                ,'Утепление стен подвала',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[typePN] == 'BRU_ОтделкаПомещений_Утеплитель_100мм'
                                ,'Утепление стен подвала',fullDf[groupModelPN])

badIsulationTypes = fullDf[fullDf[groupModelPN].isin(['Утепление цоколя штукатурного',
                   'Утепление цоколя навесного','Утепление помещений',
                   'Утепление стен подвала'])]
res = badIsulationTypes[[groupModelPN,typePN,thicknessPN]].value_counts()


#endregion
#region Обработка Монолитных фундаментных плит
fullDf = fullDf[fullDf[groupModelPN] != 'Монолитная фундаментная плита']

#endregion


#endregion
#region Экспортирование уникальных типов
uniqueTypes = fullDf[[groupModelPN,typePN,thicknessPN]].value_counts().reset_index()
if(needSaveUniqueValues):
    uniqueTypes.to_excel(uniqTypesPath,sheet_name='Sheet1')
    print('Уникальные типы сохранены')



#endregion

#endregion

#endregion
#region Сохранение очищенного датасета

if needSaveFinalDf:
    fullDf.to_csv(cleared_Path,
                  index=False,
                  sep=';')
    print('Очищенный датасет сохранен')

#endregion

#170489 rows
# res = fullDf



print(res)