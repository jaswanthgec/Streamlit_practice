import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
import numpy as np
import matplotlib.pyplot as plt
import io

# Title of the app
st.set_page_config(page_title="Analyzeapp", page_icon=":bar_chart:", layout="wide")

st.write("""
# Data Analysis, Prediction, and Visualization App

Upload an Excel or CSV file to analyze, visualize, and predict data.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'csv'])

if uploaded_file is not None:
    # Read the file into a DataFrame
    c1, c2, c3 = st.columns([3, 10, 3])
    with c2:
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
        st.write("## Enter Target Attribute")
        # Select the output attribute for prediction
        output_attribute = st.selectbox('Select the output attribute for prediction', df.columns)

        # Separate features and target variable
        X = df.drop(columns=[output_attribute])
        y = df[output_attribute]

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Handle missing values using KNNImputer for numeric columns only
        imputer = KNNImputer(n_neighbors=5)
        X_numeric = X[numeric_features]  # Select only numeric columns
        X_imputed = imputer.fit_transform(X_numeric)
        X[numeric_features] = X_imputed 

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
                'SVM': SVC(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'Neural Network': Sequential([
                    Dense(64, activation='relu', input_shape=(X_processed.shape[1],)),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid')  # Output layer for binary classification
                ])
            }

            best_score = 0
            best_model = None
            for name, model in classifiers.items():
                # Hyperparameter tuning using GridSearchCV
                param_grid = {}  # Define appropriate parameter grids for each model
                grid_search = GridSearchCV(model, param_grid, cv=5)
                grid_search.fit(X_processed, y)
                mean_score = grid_search.best_score_
                st.write(f"{name} Accuracy: {mean_score:.2f}")
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = grid_search.best_estimator_

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
                # Hyperparameter tuning using GridSearchCV
                param_grid = {}  # Define appropriate parameter grids for each model
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
                grid_search.fit(X_processed, y)
                mean_score = -grid_search.best_score_
                st.write(f"{name} RMSE: {np.sqrt(mean_score):.2f}")
                if mean_score < best_score:
                    best_score = mean_score
                    best_model = grid_search.best_estimator_

            best_model.fit(X_processed, y)
            st.write(f"Best Regression Model: {best_model.__class__.__name__} with RMSE {np.sqrt(best_score):.2f}")

        elif problem_type == 'Clustering':
            st.write("Since the output attribute is neither numeric nor categorical, clustering will be performed.")
            kmeans = KMeans(n_clusters=3)  # Example with 3 clusters
            kmeans.fit(X_processed)
            clusters = kmeans.predict(X_processed)
            st.write("KMeans Clustering Completed. Cluster assignments:")
            st.write(clusters)

            # Add visualization for clustering (e.g., scatter plot)
            # ...

        st.write("---")

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

    co1, co2 = st.columns([2, 2])
    with co1:
        if 'history' not in st.session_state:
            st.session_state.history = []
        # Allow user to choose attributes and chart type for visualization
        st.write("## Select attributes and chart type for visualization:")
        selected_attributes = st.multiselect('Select attributes (2 or more for multi-line chart)', df.columns.tolist(),
                                           default=df.columns.tolist()[:2])
        chart_type = st.selectbox('Select chart type',
                                   ['Line Chart', 'Multi-Line Chart', 'Histogram', 'Box Plot', 'Bar Chart',
                                    'Pie Chart'])

        # Filter the DataFrame based on selected attributes
        filtered_df = df[selected_attributes]
        fig, ax = plt.subplots()

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

        if st.button("Save Chart"):
            if chart_type == 'Line Chart':
                chart_data = filtered_df.to_dict()  # Store the data as a dictionary
            else:
                # For other chart types, you can store the figure as before
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                chart_data = buf

            st.session_state.history.append({
                'attributes': selected_attributes,
                'chart_type': chart_type,
                'chart_data': chart_data
            })
        st.write("---")

    with co2:
        st.write("## History for visualization:")
        st.write('This will be available if we click on the save chart button')
        for i, item in enumerate(st.session_state.history):
            col1, col2, col3 = st.columns([10, 3, 2])  # Create two columns for buttons

            with col1:
                if st.button(f"Visualization {i+1}", key=f"viz_{i}"):
                    if item['chart_type'] == 'Line Chart':
                        # Recreate the line chart from the stored data
                        df_for_chart = pd.DataFrame(item['chart_data'])
                        st.line_chart(df_for_chart)
                    else:
                        # Display other chart types as before
                        st.image(item['chart_data'],
                                 caption=f"Chart of {', '.join(item['attributes'])} ({item['chart_type']})",
                                 output_format="PNG")

            with col2:
                # Download button
                if item['chart_type'] != 'Line Chart':
                    st.download_button(
                        label="Download",
                        data=item['chart_data'],
                        file_name=f"visualization_{i+1}.png",
                        mime="image/png",
                        key=f"download_{i}"
                    )
            with col3:
                # Delete button
                if st.button("Delete", key=f"delete_{i}"):
                    del st.session_state.history[i]
                    st.experimental_rerun()

    r1, rec2, r1 = st.columns([3, 10, 3])
    with rec2:
        # Recommend charts based on data types
        st.write("## Recommended Charts:")
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
    - **LinkedIn:** [LinkedIn](https://www.linkedin.com/in/jaswanthkollipara/)
    - **Mail:** kolliparajaswanth030@gmail.com
""")
