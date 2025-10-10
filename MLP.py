import numpy as np

class MLP:
    def __init__(self):
        pass

    # Função de treinamento
    def train(self, inputs, outputs, alpha, epochs):
        self.inputs = inputs
        self.outputs = outputs
        self.alpha = alpha
        self.epochs = epochs

        # Inicialização aleatória dos pesos e bias
        w11 = np.random.uniform(0, 1)
        w12 = np.random.uniform(0, 1)
        w21 = np.random.uniform(0, 1)
        w22 = np.random.uniform(0, 1)
        wh1 = np.random.uniform(0, 1)
        wh2 = np.random.uniform(0, 1)
        b1 = np.random.uniform(0, 1)
        b2 = np.random.uniform(0, 1)
        b3 = np.random.uniform(0, 1)

        # Treinamento
        for i in range(epochs):
            for j in range(len(inputs)):
                # Camada oculta
                h1 = 1 / (1 + np.exp(-((inputs[j][0] * w11) + (inputs[j][1] * w21) + b1)))
                h2 = 1 / (1 + np.exp(-((inputs[j][0] * w12) + (inputs[j][1] * w22) + b2)))

                # Saída
                y = 1 / (1 + np.exp(-((h1 * wh1) + (h2 * wh2) + b3)))

                # Erro
                error = outputs[j][0] - y

                # Derivadas parciais
                derivative_y = y * (1 - y) * error
                derivative_h1 = h1 * (1 - h1) * wh1 * derivative_y
                derivative_h2 = h2 * (1 - h2) * wh2 * derivative_y

                # Atualização dos pesos (backpropagation simplificada)
                w11 += alpha * derivative_h1 * inputs[j][0]
                w12 += alpha * derivative_h2 * inputs[j][0]
                w21 += alpha * derivative_h1 * inputs[j][1]
                w22 += alpha * derivative_h2 * inputs[j][1]
                wh1 += alpha * derivative_y * h1
                wh2 += alpha * derivative_y * h2
                b1 += alpha * derivative_h1
                b2 += alpha * derivative_h2
                b3 += alpha * derivative_y

        # Retorna todos os pesos e bias treinados
        return w11, w12, w21, w22, wh1, wh2, b1, b2, b3

    # Função de predição
    def predict(self, weights, x1, x2):
        hidden1 = 1 / (1 + np.exp(-((x1 * weights[0]) + (x2 * weights[2]) + weights[6])))
        hidden2 = 1 / (1 + np.exp(-((x1 * weights[1]) + (x2 * weights[3]) + weights[7])))
        output = 1 / (1 + np.exp(-((hidden1 * weights[4]) + (hidden2 * weights[5]) + weights[8])))

        return 1 if output > 0.5 else 0


# ================
# TESTE DO MODELO
# ================

# Tabela verdade da porta XOR
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]

# Inicializa e treina
mlp = MLP()
trained_weights = mlp.train(inputs, outputs, alpha=0.05, epochs=10000)

# Testa todas as combinações possíveis
print("=== Teste da Porta XOR ===")
for entrada in inputs:
    x1, x2 = entrada
    result = mlp.predict(trained_weights, x1, x2)
    print(f"Entrada: {entrada} → Saída prevista: {result}")
