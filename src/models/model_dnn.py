import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import json 

def dnn(X_train, X_test, y_train, y_test):

     
    y_train = to_categorical(y_train, num_classes=11)
    y_test = to_categorical(y_test, num_classes=11)
    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_dim =X_train.shape[1]),
        #tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(11, activation='softmax')
    ])


    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    x_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    
    print('DNN modeld summary:\n')
    print(model.summary())
    
    history = model.fit(x_train, y_train, validation_split=0.5, epochs=400, batch_size=16, shuffle=True)
    
    plot_acc_name = 'acc_sequance'
    plot_loss_name = 'loss_sequance'

    if not os.path.exists(f'logs/dnn_results'):
        os.makedirs(f'logs/dnn_results')



    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'logs/dnn_results/{plot_acc_name}.jpg')

    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'logs/dnn_results/{plot_loss_name}.jpg')

    # reading the training history and save it to .json file
    new_dict = {}
    for k, v in history.history.items():
        for i in range(len(v)):    
            new_dict[i] = {}

    for k, v in history.history.items():
        for i in range(len(v)):
        #for k, v in history.history.items():
                new_dict[i][k] = v[i]

    with open(f'logs/dnn_results/train_result.json', 'w') as tr:
        json.dump(new_dict, tr, indent=4)


    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    
    with open ('logs/dnn_results/test_accuracy.csv', 'w') as f:
        f.write(str(test_acc)) 


    
