from thumos_dataset import *

import torch
import torch.nn as nn
import torch.optim as optim
import pickle

LEARNING_RATE = 0.1
EPOCHS = 200
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(device)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            nn.ReLU()
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3)),
            nn.ReLU()
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3,3)),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, len(classes)),
            #nn.Softmax(dim=1)
        )

    def forward(self, i):

        # convolutions
        x = self.conv_block_1(i)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        # final flatten & fc layer
        x = self.fc(x)

        return x

cnn_net = CNN()
if device != cpu:
    cnn_net = nn.DataParallel(cnn_net)
cnn_net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    cnn_net.parameters(),
    lr=LEARNING_RATE,
    #momentum=0.9
)

train_accuracies = []
val_accuracies = []
for e in range(EPOCHS):

    # start the background train load thread
    t = threading.Thread(target=load_data, args=(train,))
    t.start()

    losses = []
    train_correct = 0
    train_total = 0
    pbar = tqdm()
    while True:
        pbar.update(1)
        optimizer.zero_grad()

        cnn_outputs = []
        actual_labels = []
        batch = []

        done = False
        for _ in range(BATCH_SIZE):
            d = data_queue.get()

            #TODO FIX THIS END CONDITION
            if d is None:
                done = True
                break

            single_input = d[0]
            label = d[1]

            actual_labels.append(label)
            batch.append(single_input)

        # TODO FIX THIS END CONDITION
        if done:
            break

        input_tensor = torch.from_numpy(np.asarray(batch)).float()

        cnn_outputs = cnn_net(input_tensor)

        loss = criterion(cnn_outputs, torch.tensor(actual_labels).to(device).long())
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        for output, label in zip(cnn_outputs.argmax(dim=1).cpu().detach().numpy(), actual_labels):
            if output == label:
                train_correct += 1
            train_total += 1

        pbar.set_description(str(train_correct / train_total))

        del input_tensor
        del cnn_outputs
        del actual_labels

    pbar.close()

    train_accuracies.append(train_correct / train_total)
    print(f"Epoch {e} Loss: {sum(losses) / len(losses)}, Accuracy: {train_correct / train_total}")

    if not data_queue.empty():
        print("Something went wrong, result queue not empty... emptying...")
        while not data_queue.empty():
            data_queue.get()

    with torch.no_grad():

        val_correct = 0

        t = threading.Thread(target=load_data, args=(test,))
        t.start()

        pbar = tqdm()
        count = 0
        while True:
            d = data_queue.get()

            if d is None:
                break

            processed_d = torch.from_numpy(np.asarray([d[0]])).float()
            label = d[1]

            pred = cnn_net(processed_d).argmax(dim=1).item()

            if pred == label:
                val_correct += 1

            del processed_d

            pbar.update(1)

            count += 1
            pbar.set_description(str(val_correct / count))

        val_accuracies.append(val_correct / len(test))
        print(f"Epoch {e} Validation Accuract: {val_correct / len(test)}")

    print("---------------------------------------------------------------")

    torch.save({
        'epoch': e,
        'model_state_dict': cnn_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses,
    }, f"{MODEL_SAVE_DIR}/m1_{e}")

with open(f"{MODEL_SAVE_DIR}/hist", 'wb') as f:
    pickle.dump({"Train":train_accuracies, "Val":val_accuracies}, f)
