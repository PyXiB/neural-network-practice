import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Гиперпараметры
batch_size = 64
learning_rate = 0.001
epochs = 5

# Трансформации для данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Загрузка данных MNIST
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Определение архитектуры нейронной сети
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Инициализация модели, функции потерь и оптимизатора
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Функция обучения
train_losses = []
def train_model():
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Эпоха [{epoch+1}/{epochs}], Потери: {avg_loss:.4f}")

# Функция тестирования и визуализации результатов
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Точность на тестовых данных: {accuracy:.2f}%")

    # Визуализация предсказаний
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Истинное: {labels[i]} | Предсказано: {predictions[i].item()}")
        plt.axis('off')
    plt.show()

# Функция для визуализации потерь
def plot_training_loss():
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Потери на обучении')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('График потерь на обучении')
    plt.legend()
    plt.show()

# Запуск программы
if __name__ == "__main__":
    print("Обучение модели...")
    train_model()
    print("Визуализация потерь...")
    plot_training_loss()
    print("Тестирование модели...")
    test_model()


