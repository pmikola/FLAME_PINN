
def learning_loop(num_epochs,model,dataloader,criterion,optimizer,device):
    global loss
    best_loss = float('inf')  # Initialize with a very large number
    best_model_state = None
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        if (epoch + 1) % 10 == 0:
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = model.state_dict()
