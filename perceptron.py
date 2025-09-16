# import -> incorpora uma biblioteca \
# (código já escrito para resolver algum problema específico)
import numpy as np

class Perceptron:
    # Declaração do construtor da classe
    def __init__(self):
        self.weights = None

    def train(self, inputs, outputs, learning_rate=0.1, epochs=100):
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Inicialização de pesos iniciais de forma aleatória
        w1, w2, bias = np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)

        for i in range(epochs):
            for j in range(len(inputs)):
                # função de ativação
                # sigmoid = 1 / (1 + np.exp(-(w1 * inputs[j][0] + w2 * inputs[j][1] + bias)))
                output = self.activate(w1 * inputs[j][0] + w2 * inputs[j][1] + bias, activation_func='sigmoid')


                # atualização dos pesos por iteração
                w1 = w1 + learning_rate * (outputs[j][0] - output) * inputs[j][0]
                w2 = w2 + learning_rate * (outputs[j][0] - output) * inputs[j][1]
                bias = bias + (learning_rate * (outputs[j] [0] - output))

        self.weights = (w1, w2, bias)
        return self.weights

    def activate(self, z, activation_func='sigmoid'):
        if activation_func == 'sigmoid':
             return 1 / (1 + np.exp(-z))
        elif activation_func == 'step':
             return 1 if z > 0 else 0
        else:
            raise ValueError("Invalid activation function specified.")


    def predict(self, x1, x2, activation_func='sigmoid'):
        if self.weights is None:
            raise ValueError("Perceptron has not been trained yet.")
        w1, w2, bias = self.weights
        z = (x1 * w1) + (x2 * w2) + bias
        if activation_func == 'sigmoid':
            return 1 if self.activate(z, activation_func='sigmoid') > 0.5 else 0
        elif activation_func == 'step':
             return self.activate(z, activation_func='step')
        else:
            raise ValueError("Invalid activation function specified.")


if __name__ == '__main__':
    # Entradas das portas lógicas (em pares)
    inputs_or = [[0,0], [0,1], [1,0], [1,1]]
    outputs_or = [[0], [1], [1], [1]]

    inputs_and = [[0,0], [0,1], [1,0], [1,1]]
    outputs_and = [[0], [0], [0], [1]]


    print('OR Gate Test (Sigmoid Activation):')
    perceptron_or_sigmoid = Perceptron()
    weights_or_sigmoid = perceptron_or_sigmoid.train(inputs = inputs_or,
                                                     outputs = outputs_or,
                                                     learning_rate=0.1, epochs=100)

    print('Inputs  -> Output | Prediction')
    for i in range(len(inputs_or)):
        prediction = perceptron_or_sigmoid.predict(inputs_or[i][0], inputs_or[i][1], activation_func='sigmoid')
        print(f'{inputs_or[i][0]} {inputs_or[i][1]}   -> {outputs_or[i][0]}      | {prediction}')


    print('\nAND Gate Test (Sigmoid Activation):')
    perceptron_and_sigmoid = Perceptron()
    weights_and_sigmoid = perceptron_and_sigmoid.train(inputs = inputs_and,
                                                       outputs = outputs_and,
                                                       learning_rate=0.1, epochs=100)

    print('Inputs  -> Output | Prediction')
    for i in range(len(inputs_and)):
        prediction = perceptron_and_sigmoid.predict(inputs_and[i][0], inputs_and[i][1], activation_func='sigmoid')
        print(f'{inputs_and[i][0]} {inputs_and[i][1]}   -> {outputs_and[i][0]}      | {prediction}')


    print('\nOR Gate Test (Step Activation):')
    perceptron_or_step = Perceptron()
    weights_or_step = perceptron_or_step.train(inputs = inputs_or,
                                               outputs = outputs_or,
                                               learning_rate=0.1, epochs=100) # Step function training might require different parameters

    print('Inputs  -> Output | Prediction')
    for i in range(len(inputs_or)):
        prediction = perceptron_or_step.predict(inputs_or[i][0], inputs_or[i][1], activation_func='step')
        print(f'{inputs_or[i][0]} {inputs_or[i][1]}   -> {outputs_or[i][0]}      | {prediction}')


    print('\nAND Gate Test (Step Activation):')
    perceptron_and_step = Perceptron()
    weights_and_step = perceptron_and_step.train(inputs = inputs_and,
                                                 outputs = outputs_and,
                                                 learning_rate=0.1, epochs=100) # Step function training might require different parameters

    print('Inputs  -> Output | Prediction')
    for i in range(len(inputs_and)):
        prediction = perceptron_and_step.predict(inputs_and[i][0], inputs_and[i][1], activation_func='step')
        print(f'{inputs_and[i][0]} {inputs_and[i][1]}   -> {outputs_and[i][0]}      | {prediction}')


    print('\nExperimenting with Learning Rate and Epochs (OR Gate, Sigmoid):')
    learning_rates = [0.01, 0.1, 0.5]
    epochs_list = [50, 100, 500]

    for lr in learning_rates:
        for ep in epochs_list:
            print(f'\nLearning Rate: {lr}, Epochs: {ep}')
            perceptron_exp = Perceptron()
            weights_exp = perceptron_exp.train(inputs=inputs_or, outputs=outputs_or, learning_rate=lr, epochs=ep)
            correct_predictions = 0
            for i in range(len(inputs_or)):
                 prediction = perceptron_exp.predict(inputs_or[i][0], inputs_or[i][1], activation_func='sigmoid')
                 if prediction == outputs_or[i][0]:
                     correct_predictions += 1
            print(f'Accuracy: {correct_predictions / len(inputs_or)}')
