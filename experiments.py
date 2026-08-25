import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pickle


data=pd.read_csv("Churn_Modelling.csv")
print(data.head())

data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
#print(data.head())

# Gender and Geaography are in text format, we need to encode
label_gender_encoder = LabelEncoder()
data["Gender"] = label_gender_encoder.fit_transform(data["Gender"])
#print(data.head())

# Geography has more than one category, multiple different values, hence need to encode using OHE
onehot_encoder_geohot = OneHotEncoder()
geo_encoder = onehot_encoder_geohot.fit_transform(data[["Geography"]])
print(geo_encoder)
print(data.head())
print(onehot_encoder_geohot.get_feature_names_out(["Geography"]))

geo_encoded_df=pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geohot.get_feature_names_out(["Geography"]))
print(geo_encoded_df)


#combine all columns now.
data=pd.concat([data.drop("Geography", axis=1), geo_encoded_df], axis=1)
print(data.head())

#save in pickle file
with open("label_encoder_gender.pkl", "wb") as file:
    pickle.dump(label_gender_encoder, file)

with open("onehot_encoder_geohot.pkl", "wb") as file:
    pickle.dump(onehot_encoder_geohot, file)


# devide into dependent and independent
x=data.drop("Exited", axis=1)
y=data["Exited"]


# split the data in tarining and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# scale the feature
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
print(x_train)
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)



# ANN impmlementation

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime


# build ANN model

model=Sequential([
    Dense(64, activation="relu", input_shape=(x_train.shape[1],)), # hidden layer 1, connected with input layer
    Dense(32, activation="relu"), # H2
    Dense(1, activation="sigmoid") # output layer
])

print(model.summary())


# create own optimization for model
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.BinaryCrossentropy()
#compile the model

model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])


## Set up the Tensorboard
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)


## Set up Early Stopping
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)


### Train the model
history=model.fit(
    x_train,y_train,validation_data=(x_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)

model.save('model.h5')

## Load Tensorboard Extension, run this in command prompt to see logs
#%load_ext tensorboard
#%tensorboard --logdir logs/fit