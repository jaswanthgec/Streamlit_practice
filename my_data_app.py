import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import matplotlib.pyplot as plt

# Title of the app
st.write("""
# Data Analysis, Prediction, and Visualization App

Upload an Excel or CSV file to analyze, visualize, and predict data.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'csv'])

if uploaded_file is not None:
    # Read the file into a DataFrame
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except ImportError as e:
        st.error(f"Error: {e}")
        st.stop()
    
    # Display the number of attributes (columns)
    st.write(f"Number of attributes (columns): {df.shape[1]}")

    # Display the DataFrame
    st.write("DataFrame Preview:")
    st.dataframe(df.head())

    # Display column names and their data types
    st.write("Column Names and Data Types:")
    st.write(df.dtypes)
    st.write("---")

    # Select the output attribute for prediction
    output_attribute = st.selectbox('Select the output attribute for prediction', df.columns)

    # Separate features and target variable
    X = df.drop(columns=[output_attribute])
    y = df[output_attribute]

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Handle missing values
    for column in numeric_features:
        X[column].fillna(X[column].mean(), inplace=True)
    for column in categorical_features:
        X[column].fillna(X[column].mode()[0], inplace=True)

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Apply the preprocessing pipeline to the data
    X_processed = preprocessor.fit_transform(X)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Input boxes for new data
    st.write("## Enter new data for prediction:")
    new_data = {}
    for column in X.columns:
        if column in numeric_features:
            new_data[column] = st.number_input(f"Enter value for {column}", value=float(df[column].mean()))
        else:
            new_data[column] = st.selectbox(f"Select value for {column}", options=df[column].unique())

    new_data_df = pd.DataFrame([new_data])
    new_data_processed = preprocessor.transform(new_data_df)

    # Predict the output attribute
    prediction = knn.predict(new_data_processed)
    st.write(f"## Predicted {output_attribute}: {prediction[0]}")
    st.write("---")
    # Allow user to choose an attribute and chart type for visualization
    st.write("## Select an attribute and chart type for visualization:")
    selected_attribute = st.selectbox('Select attribute', df.columns)
    chart_type = st.selectbox('Select chart type', ['Line Chart', 'Histogram', 'Box Plot', 'Bar Chart', 'Pie Chart'])

    # Show filters based on the attribute type
    if pd.api.types.is_numeric_dtype(df[selected_attribute]):
        min_val, max_val = float(df[selected_attribute].min()), float(df[selected_attribute].max())
        selected_range = st.slider('Select range', min_val, max_val, (min_val, max_val))
        filtered_df = df[(df[selected_attribute] >= selected_range[0]) & (df[selected_attribute] <= selected_range[1])]
    else:
        selected_values = st.multiselect('Select values', df[selected_attribute].unique(), df[selected_attribute].unique())
        filtered_df = df[df[selected_attribute].isin(selected_values)]

    # Display the selected chart with the applied filters
    if chart_type == 'Line Chart':
        st.line_chart(filtered_df[selected_attribute])
    elif chart_type == 'Histogram':
        st.bar_chart(np.histogram(filtered_df[selected_attribute], bins=20)[0])
    elif chart_type == 'Box Plot':
        fig, ax = plt.subplots()
        ax.boxplot(filtered_df[selected_attribute].dropna())
        st.pyplot(fig)
    elif chart_type == 'Bar Chart':
        st.bar_chart(filtered_df[selected_attribute].value_counts())
    elif chart_type == 'Pie Chart':
        fig, ax = plt.subplots()
        filtered_df[selected_attribute].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
    st.write("---")
    # Recommend charts based on data types
    st.write("Recommended Charts:")
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            st.write(f"- Column `{column}` is numeric. Recommended charts: Line Chart, Histogram, Box Plot.")
            st.line_chart(df[column])
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            st.write(f"- Column `{column}` is categorical. Recommended charts: Bar Chart, Pie Chart.")
            st.bar_chart(df[column].value_counts())
        else:
            st.write(f"- Column `{column}` has an unsupported data type for automatic chart recommendations.")

# Add programmer's details at the end
st.markdown("""
    ---
    ## About the Developer
    - **Name:** Jaswanth Kollipara
    - **GitHub:** [Github](https://github.com/jaswanthgec)
    - **LinkedIn:** [LinkedIn](https://www.linkedin.com/in/jaswanth-kollipara-896443237/)
    - **Mail:** kjaswanth28@gmail.com
""")