import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier

st.title("Прогнозирование выживаемости")
st.markdown('#### Загрузите нужный csv или excel файл')

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)
    s = 1
else:
    s = 0

st.markdown('#### или введите параметры пациента вручную')


df_input = pd.DataFrame(
    [
       {'Количество метастазов': 3, "ECOG": 2, "ПКР": 1, 'Дифференцировка опухоли': 3, "Градация N": 0, "Синхронные и Метахронные метастазы": 2, 'Локализация метастазов': 1, "Наличие метастазэктомии": 0, "Число органов с метастазами": 1, 'Наличие нефрэктомии': 1, "Heng": 3}
    ]
)
edited_df_input = st.experimental_data_editor(df_input)
st.divider()
st.text(""
        "Справка по вводу данных:\n "
        "   * Ячейка 'Количество метастазов' принимает значение от 1 до 3, \nгде 1 - солитарные, 2 - единичные, 3 - множественные.\n\n"
        "   * Ячейка 'ECOG' принимает значение от 0 до 3,\n и содержит в себе кодировку статуса по ECOG.\n\n"
        "   * Ячейка 'ПКР' принимает значение от 1 до 4,\n и содержит в себе гистологический вариант ПКР.\n\n"
        "   * Ячейка 'Дифференцировка опухоли' принимает значение от 1 до 3       .\n\n"
        "   * Ячейка 'Градация N' принимает значение от 0 до 2.\n\n"
        "   * Ячейка 'Синхронные и Метахронные метастазы' принимает значение 1 или 2,\n где 1 - синхронные, 2 - метахронные.\n\n"
        "   * Ячейка 'Локализация метастазов' принимает значение от 1 до 3,\n где 1 - почка слева, 2 - почка справа, 3 - обе почки.\n\n"
        "   * Ячейка 'Наличие метастазэктомии' принимает значение от 0 до 1,\n где 0 - метастазэктомии нет, 1 - она есть.\n\n"
        "   * Ячейка 'Наличие нефрэктомии' принимает значение от 0 до 1,\n где 0 - циторедуктивной нефрэктомии нет, 1 - она есть.\n\n"
        "   * Ячейка 'Heng' принимает значение от 1 до 3,\n где 1 - благоприятная, 2 - промежуточная, 3 - неблагоприятная.\n\n"
        )

st.divider()
st.markdown('### Нажмите на кнопку готово, чтобы сделать прогноз')
form = st.form("my_form")
form.form_submit_button("Готово!")




#работа модели и вывод результата (также определить откуда полчить данные из ручного ввода или из файла)

columns_name = ['Кол-во метастазов', 'ECOG', 'ПКР', 'Дифференциовка опухоли', 'Градация N', 'Синхронные и Метахронные метастазы',
                'Локализация метастазов', 'Наличие метастазэктомии', 'Число органов с метастазами', 'Наличие нефрэктомия', 'Heng']


df_anonim = pd.read_csv('df_for_streamlit.csv')
estimator = GradientBoostingClassifier(random_state=3, learning_rate= 0.1, max_depth= 4, max_features= 1.0, min_samples_leaf= 3, n_estimators= 10)

estimator.fit(df_anonim[columns_name], df_anonim['ВПП_code'])


#Predict
y_pred = estimator.predict(X_ts_sc)









st.divider()
working_df = pd.DataFrame([])

#@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(working_df)

st.download_button(
    label="Скачать данные в формате CSV",
    data=csv,
    file_name='Canser_research.csv',
    mime='text/csv',
)



'''
X = patients[columns_name]
Y = patients['ВПП_code']

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
'''