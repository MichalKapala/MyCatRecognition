from DataStorager import DataStorage
from MainNet import Net
import matplotlib.pyplot as plt
import torch.nn as nn
import torch


def main():
    data = DataStorage()
    data.load_data_train()
    data.load_data_test()
    data.prepare_dataset()

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

    loss_tab = []
    score_tab = []
    loss_mean = 0

    for _ in range(14):
        for i in range(4, len(data.data_train), 4):
            x, label = data.data_train[i - 4:i], data.labels_train[i - 4:i]
            optimizer.zero_grad()
            predicted = net(x)
            loss = criterion(predicted, label)
            loss.backward()
            optimizer.step()
            loss_mean += loss
            if i % 100 == 0:
                loss_tab.append(loss_mean / 100)
                print(loss_tab[-1])
                loss_mean = 0

    plt.plot(range(len(loss_tab)), loss_tab)
    plt.show()

    with torch.no_grad():
        predicted_labels = []
        for i in range(4, len(data.data_test) + 4, 4):
            x = data.data_test[i - 4:i]
            predict = net(x)
            _, predict = predict.max(dim=1)
            for j in predict:
                predicted_labels.append(j.item())
        predicted_labels = torch.tensor(predicted_labels)
        score = predicted_labels != data.labels_test

    score_percentage = 100 * (data.labels_test.shape[0] - score.sum()) / data.labels_test.shape[0]
    print("Final score: ", score_percentage.item())

if __name__ == "__main__":
    main()