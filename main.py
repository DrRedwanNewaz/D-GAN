import pandas as pd
import numpy as np
import os
from keras.layers import Reshape, Flatten, LeakyReLU, Activation,LSTM
from keras.models import Sequential,load_model
from keras.optimizers import Adam
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import normal_latent_sampling, AdversarialOptimizerSimultaneous
from keras_adversarial.legacy import l1l2, Dense, fit
from dataloader import dataset

# load dataset
db = dataset(seq_len=5)



def model_generator(latent_dim, input_shape, hidden_dim=1024, reg=lambda: l1l2(1e-5, 1e-5)):
    return Sequential([
        Dense(int(hidden_dim / 4), name="generator_h1", input_dim=latent_dim, W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(int(hidden_dim / 2), name="generator_h2", W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(int(hidden_dim), name="generator_h3", W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(int(np.prod(input_shape)), name="generator_x_flat", W_regularizer=reg()),
        Activation('sigmoid'),
        Reshape(input_shape, name="generator_x")],
        name="generator")

def model_discriminator(input_shape, hidden_dim=1024, reg=lambda: l1l2(1e-5, 1e-5), output_activation="sigmoid"):

    return Sequential([
        Flatten(name="discriminator_flatten", input_shape=input_shape),
        Dense(int(hidden_dim), name="discriminator_h1", W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(int(hidden_dim / 2), name="discriminator_h2", W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(int(hidden_dim / 4), name="discriminator_h3", W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(1, name="discriminator_y", W_regularizer=reg()),
        Activation(output_activation)],
        name="discriminator")

def train_gan(adversarial_optimizer, path, opt_g, opt_d, nb_epoch, generator, discriminator, latent_dim,
                targets=gan_targets, loss='mse'):
    csvpath = os.path.join(path, "history.csv")
    id=0
    while os.path.exists(csvpath):
        name="history_%d"%id+".csv"
        csvpath = os.path.join(path, name)
        print("Already exists: {}".format(csvpath))
        id+=1


    print("Training: {}".format(csvpath))
    print (os.path.join(path, 'logs'))
    # gan (x - > yfake, yreal), z generated on GPU
    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))

    # build adversarial model
    model = AdversarialModel(base_model=gan,
                            player_params=[generator.trainable_weights, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[opt_g, opt_d],
                              loss=loss)


    xtrain, xtest = db.get_dataset(5, 'train')
    y = targets(xtrain.shape[0])
    ytest = targets(xtest.shape[0])

    history = fit(model, x=xtrain, y=y, validation_data=(xtest, ytest),  nb_epoch=nb_epoch,
                  batch_size=1,verbose=2)

    # save history to CSV
    df = pd.DataFrame(history.history)
    df.to_csv(csvpath)
    # save models
    generator.save(os.path.join(path, "generator.h5"))
    discriminator.save(os.path.join(path, "discriminator.h5"))
    return generator

def generator_sampler(G,latent_dim):
    z = np.random.normal(size=(1, latent_dim))
    return G.predict(z)

def result(G=[]):
    G=load_model('output/gan2/generator.h5')
    output=[ generator_sampler(G, latent_dim) for _ in range(1)]
    output=np.array(output)
    output=np.reshape(output,(output.shape[0]*output.shape[1],output.shape[2],output.shape[3]))
    print(np.shape(output))

    xtrain, xtest = db.get_dataset(1, 'test')
    db.viz(output,xtest)

latent_dim = 160
def main():
    input_shape = (160, 2)
    # generator (z -> x)
    generator = model_generator(latent_dim, input_shape)
    # discriminator (x -> y)
    discriminator = model_discriminator(input_shape)
    G=train_gan(AdversarialOptimizerSimultaneous(), "output/gan2",
                opt_g=Adam(1e-4, decay=1e-4),
                opt_d=Adam(1e-3, decay=1e-4),
                nb_epoch=200, generator=generator, discriminator=discriminator,
                latent_dim=latent_dim)
    return G


if __name__ == "__main__":
    G =main()
    result()
