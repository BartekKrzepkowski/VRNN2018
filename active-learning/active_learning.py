import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from livelossplot import PlotLossesKeras
from keras.callbacks import EarlyStopping, ModelCheckpoint

def train(build_model, X, Y, picker, initial_data_idx, metrices_names, model_name,
          initial_data_size=1000, steps=100, epochs=100, batch_size=128, is_ensemble=False):

    # set variables
    labeled_idx = initial_data_idx
    metrices_results = []
    x_train, x_test = X
    y_train, y_test = Y
    #set parameters
    interval = x_train.shape[0] // steps
    x_steps = np.arange(1, steps+1)
    filepath = './saved_models/best_model{}.h5'.format(model_name)
    callbacks=[
	    PlotLossesKeras(max_cols=3),
	    EarlyStopping(monitor="val_loss", patience=3),
	    ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True),
	]
    
    
    for i in tqdm(range(1, steps+1)):
        np.random.shuffle(labeled_idx)
        print("Iteration number: {}, size of training set: {}".format(i, len(labeled_idx)))
        
        #create & fit model
        model = build_model()
        model.fit(x=x_train[labeled_idx],
                  y=y_train[labeled_idx],
                  validation_data=[x_test, y_test],
                  callbacks=callbacks,
                  epochs=epochs,
                  batch_size=batch_size
                 )

        # evaluate model
        # model.load_weights(filepath) pomin wynik, indeksy
        result = model.evaluate(x_test, y_test)
        metrices_results.append(result)
        
        # whether ensemple picker or not
        # decrease uncertainty
        if labeled_idx.shape[0] != x_train.shape[0]:
            if not is_ensemble:
                labeled_idx = picker(model, x_train, labeled_idx, interval)
            else:
                labeled_idx = picker(build_model, [x_train, y_train], labeled_idx, interval)


    # plot results
    n_rows = (len(metrices_results[0])-1)//3 + 1
    n_cols = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=[20, 10])
    fig.autofmt_xdate(rotation=70)
    for i, result in enumerate(zip(*metrices_results)):
        first_idx = i//3
        second_idx = i%3
        ax[first_idx][second_idx].grid(True)
        ax[first_idx][second_idx].set_title(metrices_names[i].upper() + "_TEST")
        ax[first_idx][second_idx].set_xlabel("DATA SIZE")
        ax[first_idx][second_idx].set_ylabel(metrices_names[i])
        ax[first_idx][second_idx].set_xticks(interval * x_steps)
        ax[first_idx][second_idx].plot(x_steps * interval, result)
    
    plt.show()
    
    return metrices_results

    ##TODO
    #Uogolnic na rozne modele
    #zgrac to z initial data
    #dolozyc opcje sprawdzania wynikow najbardziej prawdopodobnych w danym kroku czasowym