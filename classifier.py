import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import matplotlib.pyplot as plt
import numpy as np

from official.nlp import optimization

import preprocess
import esg

ENCODER = "https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/2"
PREPROCESSOR = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"


def perform_assessment(cusip_path, county, state):
    """
    Main function used when the program is called using -a; performs a risk assessment using a saved model of a given
    CUSIP as a PDF and also gets National Risk Index information for the county associated with the CUSIP
    :param cusip_path: Path to the CUSIP to be assessed
    :param county: County associated with the given CUSIP
    :param state: State associated with the given CUSIP
    """
    print("Converting to text file...")
    risk_text = preprocess.process_cusip(cusip_path, 'filter.txt', 3)
    reloaded_model = tf.keras.models.load_model('./saved_model')
    result = reloaded_model(tf.constant([risk_text]))[0]

    # The labels are sorted alphabetically, therefore A appears before AA in the result list
    print(f"Probability for AAA: {result[2]:.2f}")
    print(f"Probability for AA: {result[1]:.2f}")
    print(f"Probability for A: {result[0]:.2f}")
    print(f"Probability for BAA: {result[5]:.2f}")
    print(f"Probability for BA: {result[4]:.2f}")
    print(f"Probability for B: {result[3]:.2f}")

    risk_index, risk_rating = esg.get_NRI(county, state)
    if risk_index is not None and risk_rating is not None:
        print(f"The risk index of the county is {risk_index:.2f}, which is {risk_rating.lower()}.")


def evaluate_model(testing_directory):
    print("Performing detailed evaluation...")

    test_ds = tf.keras.utils.text_dataset_from_directory(
        testing_directory,
        labels='inferred',
        label_mode='categorical',
        batch_size=1
    )

    reloaded_model = tf.keras.models.load_model('./saved_model')

    switch = {
        0: 'A',
        1: 'AA',
        2: 'AAA',
        3: 'B',
        4: 'BA',
        5: 'BAA'
    }

    index = 1
    for (text, label) in test_ds:
        print(f"[{index}/{len(test_ds)}]")
        index += 1

        result = reloaded_model(tf.constant(text))[0]

        print(f"Probability for AAA: {result[2]:.2f}")
        print(f"Probability for AA: {result[1]:.2f}")
        print(f"Probability for A: {result[0]:.2f}")
        print(f"Probability for BAA: {result[5]:.2f}")
        print(f"Probability for BA: {result[4]:.2f}")
        print(f"Probability for B: {result[3]:.2f}")

        rating = np.array(label[0]).tolist().index(1.)
        print(f"Label: {switch.get(rating)}")


def load_dataset(dataset_directory):
    """
    Loads the text file dataset to perform training and testing with; labels are inferred by directory structure
    :param dataset_directory: Path to the directory containing the data
    :return: Training, validation and testing dataset
    """

    training_directory = dataset_directory + "/training/"
    testing_directory = dataset_directory + "/testing/"

    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = 16
    SEED = 42

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        training_directory,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        subset='training',
        shuffle=True,
        seed=SEED
    )

    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        training_directory,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        subset='validation',
        shuffle=True,
        seed=SEED
    )

    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        testing_directory,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE
    )

    test_ds = raw_test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


def train_model(train_ds, val_ds, test_ds):
    """
    Builds, compiles and trains the classifier, creates a plot of the training loss and accuracy over time and then saves
    the model for inference
    :param train_ds: Training dataset
    :param val_ds: Validation dataset
    :param test_ds: Testing dataset
    """
    classifier_model = build_classifier_model()
    tf.keras.utils.plot_model(classifier_model, to_file="saved_model.png")

    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = tf.keras.metrics.CategoricalAccuracy()

    epochs = 10
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    classifier_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)

    print(f'Training model with {ENCODER}')
    history = classifier_model.fit(x=train_ds,
                                   validation_data=val_ds,
                                   epochs=epochs)

    classifier_model.save('./saved_model/', include_optimizer=False)

    history_dict = history.history

    acc = history_dict['categorical_accuracy']
    val_acc = history_dict['val_categorical_accuracy']

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.savefig("training_graph.png")

    print("Performing evaluation on model...")

    loss, accuracy = classifier_model.evaluate(test_ds)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")


def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(PREPROCESSOR,
                                         name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(ENCODER, trainable=True,
                             name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['sequence_output']
    net = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), name="Bidirectional_RNN")(net)
    net = tf.keras.layers.Dense(6)(net)
    net = tf.keras.layers.Softmax(name='Classifier')(net)
    return tf.keras.Model(inputs=text_input, outputs=net)
