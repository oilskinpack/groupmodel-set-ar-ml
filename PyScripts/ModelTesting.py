from joblib import dump,load
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np

version = 1


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
    innerWall = 'внутр' in typeName or 'перегор' in typeName
    arr.append(innerWall)

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
    concretePN = 'бетон' in typeName
    arr.append(concretePN)

    #ЛлуBool
    lluPN = 'ллу' in typeName
    arr.append(lluPN)

    #ПомещBool
    premisePN = 'помещ' in typeName
    arr.append(premisePN)

    resArr = np.array([arr])
    return resArr



#endregion
#region Загрузка файлов joblib

#Загрузка нормализатора
filePath = r'D:\Khabarov\Скрипты\6.Валидация АР\PyScalers'
fileName = fr'\normalizerAR_v{version}.save'
loaded_scaler = load(filePath+fileName)

#Загрузка модели
scalerPath = r'D:\Khabarov\Скрипты\6.Валидация АР\PyScalers'
filePath = r'D:\Khabarov\Скрипты\6.Валидация АР\PyModels'
fileName = fr'\modelAR_v{version}.joblib'
loaded_model = load(filePath+fileName)

#endregion
#region Загрузка файлов onnx

#Просмотр алгоритмов
# supp_Converters = skl2onnx.supported_converters(from_sklearn = False)

#Нормализатор
fileName = fr'\normalizerAR_v{version}.onnx'
session_normalizer = rt.InferenceSession(scalerPath+fileName)
input_name = session_normalizer.get_inputs()[0].name

#Модель
fileName = fr'\modelAR_v{version}.onnx'
session_model = rt.InferenceSession(filePath+fileName)


#endregion

#region Тестирование

# ['Толщина','ПаркингBool', 'ШтукатурныйBool',
# 'НавеснойBool', 'ВнутрBool', 'ПодвалBool', 'УтеплениеBool',
# 'ПарапетBool', 'БетонBool', 'ЛЛУBool', 'ПомещенияBool']
elem  = getInputData(typeName='BRU_ФасадНавесной_ПанельФиброцементная_140мм',
                   thickness=140,sectionName='Секция 1',
                   elev= 3000)

#Тестирование joblib
transformed_data = loaded_scaler.transform(elem)
prediction = loaded_model.predict(transformed_data)
prediction_proba = loaded_model.predict_proba(transformed_data).max()
print(f'Группа модели:{prediction}, вероятность {prediction_proba}  - joblib')

#Тестирование onnx
transformed_data_onnx = session_normalizer.run(None,{input_name:elem.astype(np.float32)})[0]
prediction_onnx = session_model.run(None,{input_name:transformed_data_onnx.astype(np.float32)})[0]
print(f'Группа модели:{prediction_onnx}  - onnx')

#endregion


# print(res)