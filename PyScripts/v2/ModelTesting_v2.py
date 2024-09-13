from joblib import dump,load
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np

version = 2
scaler_path = r'D:\Khabarov\Скрипты\6.Валидация АР\PyScalers'
scaler_save_Name = fr'\normalizerAR_v{version}.save'

model_path = r'D:\Khabarov\Скрипты\6.Валидация АР\PyModels'
model_joblib_name = fr'\modelAR_v{version}.joblib'

scaler_onnx_name = fr'\normalizerAR_v{version}.onnx'
model_onnx_name = fr'\modelAR_v{version}.onnx'


#region Методы

def getInputData(typeName,thickness,sectionName,elev):
    arr = [thickness]

    typeName = str(typeName).lower()
    sectionName = str(sectionName).lower()
    #ПаркингBool
    parkingBool = 'секц' not in sectionName
    arr.append(parkingBool)

    #ШтукатурныйBool
    plasteringPN = 'штук' in typeName
    arr.append(plasteringPN)

    #НавеснойBool
    mountedPN = 'навесн' in typeName
    arr.append(mountedPN)

    #ВнутрBool
    innerWall = 'внутр' in typeName
    arr.append(innerWall)

    #Перегородка
    partition = 'перегор' in typeName
    arr.append(partition)

    #ПодвалBool - по отметке в названии этажа
    basementPN = elev < 0
    arr.append(basementPN)

    #УтеплениеBool
    insulationPN = 'утепл' in typeName
    arr.append(insulationPN)

    #ПарапетBool
    parapetPN = 'парапет' in typeName
    arr.append(parapetPN)

    #БетонBool
    concretePN = ('бетон' in typeName) or ('жб' in typeName)
    arr.append(concretePN)

    #ЛлуBool
    lluPN = 'ллу' in typeName
    arr.append(lluPN)

    #ПомещBool
    premisePN = 'помещ' in typeName
    arr.append(premisePN)

    #Цоколь
    plinthPN = 'цокол' in typeName
    arr.append(plinthPN)

    #Зашивка
    sewingPN = 'зашивк' in typeName
    arr.append(sewingPN)

    #Обшивка
    platingPN = 'обшивка' in typeName
    arr.append(platingPN)

    #Внешняя
    externalPN = 'внешн' in typeName
    arr.append(externalPN)

    #СП
    freePN = 'свободн' in typeName
    arr.append(freePN)

    #Керамический
    ceramicPN = 'керамич' in typeName
    arr.append(ceramicPN)

    resArr = np.array([arr])
    return resArr



#endregion
#region Загрузка файлов joblib

#Загрузка нормализатора
loaded_scaler = load(scaler_path+scaler_save_Name)

#Загрузка модели
fileName = fr'\modelAR_v{version}.joblib'
loaded_model = load(model_path+model_joblib_name)

#endregion
#region Загрузка файлов onnx

#Просмотр алгоритмов
# supp_Converters = skl2onnx.supported_converters(from_sklearn = False)

#Нормализатор
session_normalizer = rt.InferenceSession(scaler_path + scaler_onnx_name)
input_name = session_normalizer.get_inputs()[0].name

#Модель
session_model = rt.InferenceSession(model_path+model_onnx_name)


#endregion

#region Тестирование

# ['Толщина','ПаркингBool', 'ШтукатурныйBool',
# 'НавеснойBool', 'ВнутрBool', 'ПодвалBool', 'УтеплениеBool',
# 'ПарапетBool', 'БетонBool', 'ЛЛУBool', 'ПомещенияBool']
elem  = getInputData(typeName='BRU_ОтделкаПомещений_МОП_Штукатурка гипсовая по утеплителю_15мм',
                   thickness=15,sectionName='Секция 1',
                   elev= 3000)

#Тестирование joblib
transformed_data = loaded_scaler.transform(elem)
print(transformed_data)
prediction = loaded_model.predict(transformed_data)
prediction_proba = loaded_model.predict_proba(transformed_data).max()
print(f'Группа модели:{prediction}, вероятность {prediction_proba}  - joblib')

#Тестирование onnx
transformed_data_onnx = session_normalizer.run(None,{input_name:elem.astype(np.float32)})[0]
prediction_onnx = session_model.run(None,{input_name:transformed_data_onnx.astype(np.float32)})[0]
print(f'Группа модели:{prediction_onnx}  - onnx')

#endregion