import os

def evaluate(train_data_provider, val_data_provider, model, configs):

    loss , CER_train, WER_train= model.evaluate(train_data_provider) 
    print(f'MFCC CER on training data: {CER_train} \nMFCC WER on training data: {WER_train} \nMFCC Loss on training data: {loss}','\n') 

    loss , CER_test, WER_test= model.evaluate(val_data_provider) 
    print(f'MFCC CER on testing data: {CER_test} \nMFCC WER on testing data: {WER_test} \nMFCC Loss on testing data: {loss}','\n') 

    model_path = os.path.join(configs.model_path, "model_100epochs.keras")
    model = model.save(model_path)

    return model