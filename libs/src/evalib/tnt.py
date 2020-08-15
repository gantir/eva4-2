import torch
from evalib import utils
from tqdm import tqdm


def l1_penalty(x):
    # L1 regularization adds an L1 penalty equal
    # to the absolute value of the magnitude of coefficients
    return torch.abs(x).sum()


def train(model, data_loader, criterion, optimizer, l1_decay=1e-3):
    device = utils.get_device()
    model.train()

    train_acc = 0
    train_loss = 0

    correct = 0

    pbar = tqdm(data_loader)

    for batch_idx, (inputs, targets) in enumerate(pbar):

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do
        # backpropragation because PyTorch accumulates the gradients on subsequent
        # backward passes.

        # Because of this, when you start your training loop, ideally you should
        # zero out the gradients so that you do the parameter update correctly.

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # @todo: fix how l1 loss works
        # l1_loss = 0
        # if 0 < l1_decay:
        #     for param in model.parameters():
        #         l1_loss += torch.norm(param, 1)

        #     l1_loss = l1_decay * l1_loss
        # loss += l1_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        _, pred = torch.max(outputs.data, 1)
        correct += (pred == targets).sum().item()

        train_acc = 100.0 * correct / len(data_loader.dataset)

        output_message = "".join(
            ("Batch Id/Size: {}/{}, Training Loss: {:.8f}, ", "Training Accuracy: {:.4f}%",)
        ).format(batch_idx + 1, len(data_loader.dataset), train_loss, train_acc)
        pbar.set_description(desc=output_message)

    return train_acc, train_loss


def test(model, data_loader, criterion):
    device = utils.get_device()
    model.eval()

    test_acc = 0
    test_loss = 0

    correct = 0

    with torch.no_grad():
        pbar = tqdm(data_loader)

        for _batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, pred = outputs.max(1)
            correct += (pred == targets).sum().item()

        test_loss /= len(data_loader.dataset)
        test_acc = 100.0 * correct / len(data_loader.dataset)

        output_message = "".join(("\nBatch Id/Size: {}/{}, Test Loss: {:.8f}, ", "Test Accuracy: {:.4f}%\n")).format(
            _batch_idx + 1, len(data_loader.dataset), test_loss, test_acc
        )

        print(output_message)

    return test_acc, test_loss


def train_n_test(
    model, criterion, optimizer, scheduler, train_data_loader, test_data_loader, num_epochs=10, l1_decay=1e-3,
):
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []

    for epoch in range(1, num_epochs + 1):
        print("Epoch: {}".format(epoch))
        train_acc, train_loss = train(model, train_data_loader, criterion, optimizer, l1_decay)
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)

        val_acc, val_loss = test(model, test_data_loader, criterion)
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)

        if scheduler is not None:
            try:
                print("LR:", scheduler.get_lr())
            except Exception:
                pass
            scheduler.step(val_loss)

    return [
        (train_acc_history, train_loss_history),
        (val_acc_history, val_loss_history),
    ]


def get_correct_wrong_predictions(model, data_loader, count=10):
    device = utils.get_device()
    wrong_classification = []
    correct_classification = []

    for images, labels in iter(data_loader):
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        _, pred = output.max(1)
        is_correct = pred == labels

        wrong_classification_index = (is_correct == 0).nonzero().cpu().numpy()[:, 0]
        for idx in wrong_classification_index:
            if count <= len(wrong_classification):
                break
            wrong_classification.append(
                {"target": labels[idx].cpu().numpy(), "pred": pred[idx].cpu().numpy(), "img": images[idx].cpu(),}
            )

        correct_classification_index = (is_correct == 1).nonzero().cpu().numpy()[:, 0]
        for idx in correct_classification_index:
            if count <= len(correct_classification):
                break
            correct_classification.append(
                {"target": labels[idx].cpu().numpy(), "pred": pred[idx].cpu().numpy(), "img": images[idx].cpu(),}
            )

        if count <= len(correct_classification) and count <= len(wrong_classification):
            break

    return correct_classification, wrong_classification
