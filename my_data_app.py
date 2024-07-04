import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
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
    st.write("## Select the Target Attribute")
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

    # Determine the problem type
    if pd.api.types.is_numeric_dtype(y):
        problem_type = 'Regression'
    elif pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y):
        problem_type = 'Classification'
    else:
        problem_type = 'Clustering'

    st.write(f"Determined problem type: {problem_type}")

    # Evaluate models and select the best one
    if problem_type == 'Classification':
        classifiers = {
            'KNN': KNeighborsClassifier(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC()
        }

        best_score = 0
        best_model = None
        for name, model in classifiers.items():
            scores = cross_val_score(model, X_processed, y, cv=5)
            mean_score = scores.mean()
            st.write(f"{name} Accuracy: {mean_score:.2f}")
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
        
        best_model.fit(X_processed, y)
        st.write(f"Best Classification Model: {best_model.__class__.__name__} with accuracy {best_score:.2f}")

    elif problem_type == 'Regression':
        regressors = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'SVR': SVR()
        }

        best_score = float('inf')
        best_model = None
        for name, model in regressors.items():
            scores = cross_val_score(model, X_processed, y, cv=5, scoring='neg_mean_squared_error')
            mean_score = -scores.mean()
            st.write(f"{name} RMSE: {np.sqrt(mean_score):.2f}")
            if mean_score < best_score:
                best_score = mean_score
                best_model = model
        
        best_model.fit(X_processed, y)
        st.write(f"Best Regression Model: {best_model.__class__.__name__} with RMSE {np.sqrt(best_score):.2f}")

    elif problem_type == 'Clustering':
        st.write("Since the output attribute is neither numeric nor categorical, clustering will be performed.")
        kmeans = KMeans(n_clusters=3)  # Example with 3 clusters
        kmeans.fit(X_processed)
        clusters = kmeans.predict(X_processed)
        st.write("KMeans Clustering Completed. Cluster assignments:")
        st.write(clusters)
    st.write("----")

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
    if problem_type in ['Classification', 'Regression']:
        prediction = best_model.predict(new_data_processed)
        st.write(f"## Predicted {output_attribute}: {prediction[0]}")
    st.write("---")

    # Allow user to choose attributes and chart type for visualization
    st.write("## Select attributes and chart type for visualization:")
    selected_attributes = st.multiselect('Select attributes (2 or more for multi-line chart)', df.columns.tolist(), default=df.columns.tolist()[:2])
    chart_type = st.selectbox('Select chart type', ['Line Chart', 'Multi-Line Chart', 'Histogram', 'Box Plot', 'Bar Chart', 'Pie Chart'])

    # Filter the DataFrame based on selected attributes
    filtered_df = df[selected_attributes]

    # Display the selected chart with the selected attributes
    if chart_type == 'Line Chart':
        st.line_chart(filtered_df)
    elif chart_type == 'Multi-Line Chart' and len(selected_attributes) > 1:
        fig, ax = plt.subplots()
        for attribute in selected_attributes:
            ax.plot(filtered_df.index, filtered_df[attribute], label=attribute)
        ax.legend()
        st.pyplot(fig)
    elif chart_type == 'Histogram':
        for attribute in selected_attributes:
            fig, ax = plt.subplots()
            ax.hist(filtered_df[attribute], bins=20)
            ax.set_title(attribute)
            st.pyplot(fig)
    elif chart_type == 'Box Plot':
        fig, ax = plt.subplots()
        ax.boxplot([filtered_df[attribute].dropna() for attribute in selected_attributes], labels=selected_attributes)
        st.pyplot(fig)
    elif chart_type == 'Bar Chart':
        for attribute in selected_attributes:
            fig, ax = plt.subplots()
            filtered_df[attribute].value_counts().plot.bar(ax=ax)
            ax.set_title(attribute)
            st.pyplot(fig)
    elif chart_type == 'Pie Chart' and len(selected_attributes) == 1:
        fig, ax = plt.subplots()
        filtered_df[selected_attributes[0]].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
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
