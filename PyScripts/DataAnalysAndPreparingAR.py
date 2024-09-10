import re
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl



res = ''
needSaveFinalDf = False
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

def unionAllDfs(directory):
    dfList = []
    fullDf = np.NaN
    files = os.listdir(directory)
    for file in files:
        fileName = fr'\{file}'
        fullPath = directory + fileName
        df = pd.read_csv(fullPath,sep=';')
        dfList.append(df)

    fullDf = pd.DataFrame(columns=dfList[0].columns)

    for df in dfList:
        fullDf = pd.concat([fullDf,df],sort=False,axis=0)
    return fullDf

def createCountPlot(savePath,saveName,testDf,hueParam):
    fig = plt.figure(dpi=300)
    ax = fig.add_axes([0.25, 0.25, 0.7, 0.7])
    ax = sns.countplot(data=testDf, y=groupModelPN, hue=hueParam, orient='h')
    plt.yticks(rotation=45)
    ax.set_xlim(right=300, left=0)
    plt.savefig(savePath + saveName)
    fig.clf()


def createPlots(savePath,dfListWithDoubThickGM):
    for df in dfListWithDoubThickGM:
        thick = df[thicknessPN].unique()[0]


        # Рисуем график Толщина - Штукатурный
        saveName = fr'\countPlotByThickAndPlast - {thick}.jpg'
        createCountPlot(savePath, saveName, df,plasteringPN)

        # Рисуем график Толщина - Навесной
        saveName = fr'\countPlotByThickAndMounted - {thick}.jpg'
        createCountPlot(savePath, saveName, df,mountedPN)

        # Рисуем график Толщина - ВнутрСтена
        saveName = fr'\scatterPlotByThickAndInnerWall - {thick}.jpg'
        createCountPlot(savePath, saveName, df,innerWall)

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

#region Анализ и чистка ГруппыМодели (целевая функция)

dir = r'/Csv'
fullDf = unionAllDfs(dir)


#Уникальные группы модели
uniqueGM = fullDf[groupModelPN].unique()
# res = uniqueGM

#Смотрим какие группы модели не проходят по стандарту
fullDf[gmCorrect] = np.where(fullDf[groupModelPN].isin(groupModelNames),True,False)
gmNotMapped = fullDf[fullDf[gmCorrect] == False] [groupModelPN].unique()
# res = gmNotMapped

#Маппим странные группы модели
fullDf[groupModelPN] = np.where(fullDf[gmCorrect] == True,fullDf[groupModelPN],fullDf[groupModelPN].map(gmErrorMap))
uniqueGM = fullDf[groupModelPN].unique()
# res = uniqueGM

#Чистим неправильные группы модели
fullDf[groupModelPN] = np.where(fullDf['Тип'].str.contains('аквапанель',case=False) == True,'Навесной фасад',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf['Тип'] == 'BRU_Перегородка_БлокКерамический_250мм'
                                ,'Внутренняя стена. Кладка',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where(fullDf[thicknessPN].astype(float) == 300
                                ,'Монолитная фундаментная плита',fullDf[groupModelPN])



#Удаляем данные где нет ГруппыМодели - 47258
fullDf = fullDf.dropna(subset=[groupModelPN])
#res = len(fullDf)

#endregion
#region Анализ числовых признаков и заполнение нулевых значений

#Смотрим длины - 47258 non-null  float64
fullDf[lengthPN] = fullDf[lengthPN].astype(float)
# res = fullDf[lengthPN].info()

#Смотрим толщины - 47099 non-null  float64
fullDf[thicknessPN] = fullDf[thicknessPN].astype(float)
# res = fullDf[thicknessPN].info()
#Убираем строки с нулевым значением толщины
# fullDf = fullDf.dropna(subset=[thicknessPN])
fullDf[thicknessPN] = fullDf[thicknessPN].fillna(0)
# res = len(fullDf)

#Смотрим высоты - 47099 non-null
fullDf[heightPN] = fullDf[heightPN].astype(float)
# res = fullDf[heightPN].info()

#Смотрим объем - 47099 non-null
# fullDf[volumePN] = fullDf[volumePN].astype(float)
fullDf[volumePN] = fullDf[volumePN].fillna(0)
# res = fullDf[volumePN].info()

#endregion
#region Конструирование признаков

#Добавляем условие Паркинг(bool)
fullDf[parkingPN] = np.where(fullDf[sectionPN].str.contains('Секц') == True,False,True)
res = fullDf[fullDf[parkingPN] == True][groupModelPN].value_counts()

#Добавляем условие Штукатурный
fullDf[plasteringPN] = np.where(fullDf['Тип'].str.contains('штук',case=False) == True,True,False)

#Добавляем условие Навесной
fullDf[mountedPN] = np.where(fullDf['Тип'].str.contains('навесн',case=False) == True,True,False)

#Добавляем условие Перегородка или Внутренняя
fullDf[innerWall] = np.where((fullDf['Тип'].str.contains('перегор',case=False) == True)
                             | (fullDf['Тип'].str.contains('внутр',case=False) == True)
                             ,True,False)


#endregion
#region Проверка достаточности признаков

#Какие элементы можем спутать по толщине - получаем датафрейм
# Толщина  Группа модели
# 10.0     Штукатурка фасадная                            3215
# 12.5     Навесной фасад                                   24
# 15.0     Штукатурка фасадная                             336
#          Штукатурка черновая                            2440
#===Сгруппированный список по толщинам и ГМ
gmAndThickValues = fullDf.groupby([thicknessPN,groupModelPN]).size()

# 0     15.0
# 1     50.0
# 2    100.0
#Толщины по которым есть несколько групп модели
doubledThicknesses = gmAndThickValues.reset_index().groupby(thicknessPN).size() \
    [gmAndThickValues.reset_index().groupby(thicknessPN).size() > 1].reset_index()[thicknessPN]

#     Толщина                                Группа модели     0
# 2      15.0                          Штукатурка фасадная   336
# 3      15.0                          Штукатурка черновая  2440
# 5      50.0                          Утепление помещений     5
#Плоский ДФ с толщинами и ГМ (только по тем, где есть несколько ГМ на одну толщину)
dfWithDoubThickGM = gmAndThickValues.reset_index()[gmAndThickValues.reset_index()[thicknessPN].isin(doubledThicknesses)]

dfListWithDoubThickGM = []
for thick in doubledThicknesses:
    filDf = fullDf[fullDf[thicknessPN] == thick]
    dfListWithDoubThickGM.append(filDf)

testDf = dfListWithDoubThickGM[0]


#Создание графиков
savePath = r'D:\Khabarov\Скрипты\6.Валидация АР\Аналитика'
# createPlots(savePath, dfListWithDoubThickGM)

#endregion
#region Чистка признаков и добавление дополнительных

#Чистим значения - Утепление помещений только выше 0
fullDf['Отметка'] = fullDf[floorPN].apply(getElev)
fullDf[floorPN] = np.where( (fullDf[groupModelPN] == 'Утепление помещений') & (fullDf['Отметка'] < 0),
                            'Этаж 01 (отм. +0,000)',
                            fullDf[floorPN])


#Добавляем параметр ПодвалBool
fullDf['Отметка'] = fullDf[floorPN].apply(getElev)
fullDf[basementPN] = np.where(fullDf['Отметка'] >= 0,False,True)
# res = fullDf[[floorPN,basementPN]].value_counts()

#Чистим значения - Утепление штукатурного фасада
res = fullDf[fullDf[groupModelPN] == 'Утепление штукатурного фасада'] [['Тип',plasteringPN]].value_counts()
fullDf[groupModelPN] = np.where( (fullDf['Тип'] == 'BRU_ВнешняяСтена_Тех.Этаж_Утеплитель_100мм') & (fullDf[basementPN] == True),
                                 'Утепление стен подвала',fullDf[groupModelPN] )
fullDf[groupModelPN] = np.where( fullDf['Тип'] == 'BRU_ФасадНавесной_Утеплитель_100мм',
                                'Утепление навесного фасада',fullDf[groupModelPN] )
fullDf[groupModelPN] = np.where( (fullDf['Тип'] == 'BRU_ОтделкаПомещений_Утеплитель_100мм') & (fullDf['Отметка'] > 0),
                                'Утепление помещений',fullDf[groupModelPN] )
# res = fullDf[fullDf[groupModelPN] == 'Утепление штукатурного фасада'] [['Тип',plasteringPN]].value_counts()
fullDf[groupModelPN] = np.where( (fullDf['Тип'] == 'BRU_ФасадШтукатурный_Утеплитель_150мм_Цоколь'),
                                'Утепление штукатурного фасада',fullDf[groupModelPN])
fullDf[groupModelPN] = np.where( (fullDf['Тип'] == 'BRU_ФасадШтукатурный_Утеплитель_150мм')& (fullDf[groupModelPN] == 'Утепление помещений'),
                                'Утепление штукатурного фасада',fullDf[groupModelPN])

#Чистим значения Утепление цоколя
# fullDf[groupModelPN] = np.where( (fullDf[groupModelPN] == 'Утепление цоколя') & (fullDf[basementPN] == True),
#                                  'Утепление стен подвала',fullDf[groupModelPN] )


#Добавляем условие Утеплитель
fullDf[insulationPN] = np.where(fullDf['Тип'].str.contains('Утепл',False) == True,True,False)
# res =fullDf[fullDf[insulationPN] == True] [['Тип',insulationPN,groupModelPN]].value_counts()
fullDf[parapetPN] = np.where(fullDf['Тип'].str.contains('Парапет',False) == True,True,False)

#Добавляем условие Бетон
fullDf[concretePN] = np.where(fullDf['Тип'].str.contains('Бетон',False) == True,True,False)
#Добавляем условие ЛЛУ
fullDf[lluPN] = np.where(fullDf['Тип'].str.contains('ЛЛУ',False) == True,True,False)
#Добавляем условие Помещений
fullDf[premisePN] = np.where(fullDf['Тип'].str.contains('Помещ',False) == True,True,False)

#endregion
#region Проверка уникальности и достаточности финальных признаков

# res = fullDf[x_colums].drop_duplicates()
notUniqueFeatures =fullDf[x_colums_withGM].drop_duplicates().groupby(x_colums).size() \
            [fullDf[x_colums_withGM].drop_duplicates().groupby(x_colums).size() >1]
# allFeaturesByThick = fullDf[fullDf[thicknessPN] ==150] [x_colums_withGM]\
#                 .value_counts()\
#                 .reset_index()

# res = allFeaturesByThick.groupby(x_colums,as_index=False)[groupModelPN].agg(','.join)
# res = fullDf[fullDf[groupModelPN] == 'Утепление помещений'] [['Отметка',floorPN,basementPN]].value_counts()
# res = fullDf[(fullDf[groupModelPN] == 'Утепление помещений')] [['Тип',basementPN,groupModelPN]].value_counts()

#Проверка уникальности значений
#87
res = len(fullDf[x_colums_withGM].value_counts())
#87
res = len(fullDf[x_colums].value_counts())

#График распределения внутри толщин по числовым величинам
# allthick = fullDf[thicknessPN].unique()
# for thick in allthick:
#     thickDf = fullDf[fullDf[thicknessPN] == thick]
#     fig = plt.figure(dpi=300)
#     ax = fig.add_axes([0.25, 0.25, 0.7, 0.7])
#     ax = sns.scatterplot(data=thickDf, y=groupModelPN, x=heightPN)
#     plt.yticks(rotation=45)
#     # ax.set_xlim(right=300, left=0)
#     saveName = fr'\scatterplotPlotByHeight - {thick}.jpg'
#     plt.savefig(savePath + saveName)
#     fig.clf()

# res = fullDf[['Тип',groupModelPN]].value_counts().reset_index()


#endregion

#endregion


#region Сохранение финального датасета

#Сохранение модели
if needSaveFinalDf:
    fullDf.to_csv(fr'D:\Khabarov\Скрипты\6.Валидация АР\Промежуточные датасеты\dataset_ar_cleared_v{version}.txt',
                  index=False,
                  sep=';')


#endregion


print(res)
# plt.show()