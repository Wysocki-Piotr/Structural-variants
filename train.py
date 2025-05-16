import torch
from torch import nn



def train_image_model(
        model,
        dataset,
        n_epochs=10,
        batch_size=32,
        learning_rate=0.001
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to('cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cpu'), labels.to('cpu')

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()


            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {running_loss / len(dataloader)}")




