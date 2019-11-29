import pandas as pd
import numpy as np
import keras
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

#importing dataset
df = pd.read_csv('~/Desktop/project3/new_code/famdtrain.csv')
ans = pd.read_csv('~/Desktop/project3/new_code/conv_gname.csv')
df.fillna('0', inplace=True)
#df['gname'] = df['gname'].astype('category')
#df['new_gname'] = df['gname'].cat.codes

#converting to matrix
df = df.values

#converting non numerical data
def handle_non_numerical_data(ans):
    columns = ans.columns.values

    for column in columns:
        text_to_digit = {}

        def convert_to_int(val):
            return text_to_digit[val]

        if ans[column].dtype != np.int64 and ans[column].dtype != np.float64:
            column_content = ans[column].values.tolist()
            unique_elements = set(column_content)
            b=0
            for unique in unique_elements:
                if unique not in text_to_digit:
                    text_to_digit[unique] = b
                    b+=1

            ans[column] = list(map(convert_to_int, ans[column]))
            fo = open('reference.txt','w')
            for k, v in text_to_digit.items():
                fo.write(str(k) + '>>>' +  str(v) + '\n\n')
            #print(text_to_digit)
    return ans

ans = handle_non_numerical_data(ans)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(ans)
encoded_Y = encoder.transform(ans)

# convert integers to dummy variables (i.e. one hot encoded)
Y = np_utils.to_categorical(encoded_Y)

#split model to train and test
X1,X2,Y1,Y2 = train_test_split(df,Y, test_size=0.1173, random_state = 0)
#X1 = df[:7674]
#Y1 = ans[:7674]
#X2 = df[7674:]
#Y2 = ans[7674:]

# create model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=21))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(227, activation='softmax'))

#Y3 = keras.utils.to_categorical(Y1, num_classes=227)
#Y4 = keras.utils.to_categorical(Y2, num_classes=227)

#optimizer for multi class classification
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compile model
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
model.fit(X1, Y1,validation_data=(X2,Y2), epochs=20, batch_size=1280)

#evaluate the model
scores = model.evaluate(X2, Y2)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print('{} is {} '.format(model.metrics_names[0],scores[0]))

#prediction for 2016 data
pred = model.predict_classes(X2, batch_size=51, verbose=0)
for i in pred:
    print(i)

#probabilty prediction
pred1 = model.predict_proba(X2,batch_size=51,verbose=0)
print(pred1)
