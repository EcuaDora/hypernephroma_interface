import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import base64

columns_name = ['Кол-во метастазов', 'ECOG', 'ПКР', 'Дифференциовка опухоли', 'Градация N', 'Синхронные и Метахронные метастазы',
                'Локализация метастазов', 'Наличие метастазэктомии', 'Число органов с метастазами', 'Наличие нефрэктомия', 'Heng']
df_anonim = pd.read_csv('df_for_streamlit.csv')
model = GradientBoostingClassifier(random_state=3, learning_rate=0.1, max_depth=4, max_features=1.0,
                                               min_samples_leaf=3, n_estimators=10)
# Загрузка данных из CSV файла
def load_data_from_csv(file):
    data_pd = pd.read_csv(file)
    data = data_pd[columns_name]

    return data

def save_model(model):
    joblib.dump(model, 'model.pkl')
# Ввод данных вручную
def enter_data_manually():
    data = pd.DataFrame()
    for i in range(0, 11):
        feature = st.number_input(f'{columns_name[i]}:')
        data.loc[0, f'{columns_name[i]}'] = feature
    return data


# Заголовок страницы
st.title('Survival Predictions')
st.divider()
st.markdown('__Справка по вводу данных:__')
st.markdown("- Ячейка __'Количество метастазов'__ принимает значение от 1 до 3, где 1 - солитарные, 2 - единичные, 3 - множественные.")
st.markdown("- Ячейка __'ECOG'__ принимает значение от 0 до 3, и содержит в себе кодировку статуса по ECOG.")
st.markdown("- Ячейка __'ПКР'__ принимает значение от 1 до 4, и содержит в себе гистологический вариант ПКР.")
st.markdown("- Ячейка __'Дифференцировка опухоли'__ принимает значение от 1 до 3.")
st.markdown("- Ячейка __'Градация N'__ принимает значение от 0 до 2.")
st.markdown("- Ячейка __'Синхронные и Метахронные метастазы'__ принимает значение 1 или 2, где 1 - синхронные, 2 - метахронные.")
st.markdown("- Ячейка __'Локализация метастазов'__ принимает значение от 1 до 3, где 1 - почка слева, 2 - почка справа, 3 - обе почки.")
st.markdown("- Ячейка __'Наличие метастазэктомии'__ принимает значение от 0 до 1,где 0 - метастазэктомии нет, 1 - она есть.")
st.markdown("- Ячейка __'Наличие нефрэктомии'__ принимает значение от 0 до 1, где 0 - циторедуктивной нефрэктомии нет, 1 - она есть.")
st.markdown("- Ячейка __'Heng'__ принимает значение от 1 до 3, где 1 - благоприятная, 2 - промежуточная, 3 - неблагоприятная.")
st.divider()
# Выбор источника данных
data_source = st.sidebar.selectbox('Select data source', ('CSV File', 'Manual Input'))





# Загрузка модели Gradient Boosting
def load_model():
    model = joblib.load('model.pkl')
    return model



# Предсказание на данных
def predict(model, data):
    predictions = model.predict(data)
    return predictions




# Загрузка данных
data = None
if data_source == 'CSV File':
    csv_file = st.file_uploader('Upload CSV file', type=['csv'])
    if csv_file is not None:
        data = load_data_from_csv(csv_file)
else:
    data = enter_data_manually()

# Вывод данных
if data is not None:
    st.subheader('Input Data')
    st.write(data)

    # Загрузка или создание модели
    if st.button('Train Model'):
        model = GradientBoostingClassifier(random_state=3, learning_rate=0.1, max_depth=4, max_features=1.0,
                                           min_samples_leaf=3, n_estimators=10)

        X_train = df_anonim[columns_name]
        y_train = df_anonim['ВПП_code']
        model.fit(X_train, y_train)
        save_model(model)
        st.write('Model trained and saved!')
    else:
        model = load_model()

    # Проверка, что модель обучена
    if 'model' in locals() and isinstance(model, GradientBoostingClassifier):
        # Предсказание
        if st.button('Predict'):
            predictions = predict(model, data)
            st.divider()
            st.subheader('Predictions')
            st.write(predictions)
            st.markdown('__Что означают цифры:__')
            st.markdown("- __'0'__ означает, что пациент проживет ориентировочно 5 лет и больше.")
            st.markdown("- __'1'__ означает, что пациент проживет ориентировочно 1 год и меньше.")
            st.markdown("- __'2'__ означает, что пациент проживет ориентировочно от 1 года до 2 лет.")
            st.markdown("- __'3'__ означает, что пациент проживет ориентировочно от 2 до 3 лет.")
            st.markdown("- __'4'__ означает, что пациент проживет ориентировочно от 3 до 5 лет.")

            predictions_df = data.copy()
            predictions_df['Prediction'] = predictions


            # Скачивание датасета
            csv = predictions_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions</a>'
            st.markdown(href, unsafe_allow_html=True)


    else:
        st.write('Please train the model first.')











