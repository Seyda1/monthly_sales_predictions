#  Monthly Sales Prediction

## Objective

The goal of this case study is to **predict monthly sales** for each (`shop_id`, `item_id`) pair using time series forecasting methods.

## File Structure

Here's the structure of the project:
```
├── 
├── catboost_info/
├── data/
├── models/
│   ├── catboost_model.pkl
│   └── lstm_model.h5
├── notebooks/
│   ├── Data_Analysis.ipynb
│   ├── Dataset_Creation.ipynb
│   ├── monthly_sales_cb.csv
│   ├── testdataprep.ipynb
│   ├── Training_Model.ipynb
│   └── Tuning.ipynb
├── python_files/
│   ├── __pycache__/
│   ├── config.py
│   ├── createdb.py
│   ├── helpers.py
│   ├── predict.py
│   ├── test_process_lstm.py
│   └── testpreprocess.py
├── .env
├── app.py
├── db.py
├── settings.py
```


## Improvements  

Here are some areas where future improvements can be made:  

- **Trying More Models** : I could add a third model like Facebook Prophet to compare with CatBoost and LSTM. Also, having a simple baseline model would help to see how much better the others are. Since this is a regression problem, RMSE is a good metric for comparison, but considering others like MAE (Mean Absolute Error) or R² could give a more complete understanding of model performance.

- **More Hyperparameter Optimization**: More experiments can be run using Optuna. Each model can be trained with different sets of parameters multiple times to get more reliable results.

- **Feature Importance & SHAP**:Using SHAP values or feature importance scores can help explain which features have the most impact on the models’ predictions. This not only improves interpretability but also guides feature selection.

- **More Feature Engineering**: There’s potential to do more feature engineering to create or select better features. Improved features often lead to better model results, sometimes even more than tweaking the model itself.
  
- **Data Storage in Cloud Database**:Source and processed data can be saved to cloud storage like AWS RDS or Redshift. The application can be updated to read data directly from these sources.

- **Model Versioning with MLflow**: MLflow can be used for model registry and versioning, allowing continuous improvements to the model without affecting the API's functionality. This ensures that new model versions can be deployed without downtime or breaking the API.

- **Scalable Deployment with AWS EC2**: The API can be deployed to an AWS EC2 instance, providing a scalable, production-ready environment for handling more traffic and data.

- **Docker Compose for Integration**: Docker Compose can be used to orchestrate all the containers (API, database, ML models, etc.), making it easier to set up and maintain all components together.

- **CI/CD Pipeline with GitLab CI**: Implementing a CI/CD pipeline using a `gitlab-ci.yaml` file would automate testing, building, and deployment, ensuring smoother releases and updates to the project.

- **Model Accuracy**: Explore more advanced machine/deep learning models or hyperparameter tuning to improve the prediction accuracy.

- **Error Handling**: Implement better error handling and validation for edge cases or unexpected input.

- **Scalability**: Improve the scalability of the API to handle larger datasets or more concurrent users.

- **Documentation**: Expand documentation to cover additional use cases and setup instructions for different environments.

- **Unit Testing**: Add comprehensive unit tests to ensure API reliability and code quality.

- **User Interface**: Develop a user interface for interacting with the API for non-technical users.
