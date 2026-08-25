import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

### Load the trained model, scaler pickle,onehot
model=load_model('model.h5')

## load the encoder and scaler
with open('onehot_encoder_geohot.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Example input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}


# One-hot encode 'Geography'
geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))
print(geo_encoded_df)


input_df=pd.DataFrame([input_data])
print(input_df)


## Encode categorical variables
input_df['Gender']=label_encoder_gender.transform(input_df['Gender'])
print(input_df['Gender'])


## concatination one hot encoded 
input_df=pd.concat([input_df.drop("Geography",axis=1),geo_encoded_df],axis=1)
input_df


## Scaling the input data
input_scaled=scaler.transform(input_df)
print(input_scaled)


# prediction

prediction = model.predict(input_scaled)
print(prediction)


prediction_proba = prediction[0][0]


if prediction_proba > 0.5:
    print('The customer is likely to churn.')
else:
    print('The customer is not likely to churn.')