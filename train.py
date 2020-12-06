import os

from sklearn.utils import shuffle
import torch
from torch.utils.tensorboard import SummaryWriter


def train_epoch(net, criterion, optimizer, data, batch_size, batch_no):
    data = shuffle(data)
    total_loss = 0
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        x_var = torch.FloatTensor(data[start:end])
        x_var = x_var.to(net.device)
        net.to(net.device)

        optimizer.zero_grad()
        xpred_var = net(x_var)
        loss = criterion(xpred_var, x_var)
        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / (batch_size * batch_no)


def log_activation_data(net, activation_writers, X_test, Y_test, num_stripes, epoch):
    stripe_stats = net.get_stripe_stats(X_test, Y_test)
    for stripe in range(num_stripes):
        stripe_writer = activation_writers[stripe]
        for digit in range(10):
            stripe_writer.add_scalar(f'digit_{digit}', stripe_stats[digit][stripe], epoch)
        stripe_writer.flush()


def train(net,
          criterion,
          optimizer,
          log_path,
          X_train,
          X_test,
          Y_test,
          num_stripes,
          num_epochs,
          batch_size,
          batch_no,
          distort_prob_decay,
          relax_stripe_sparsity,
          log_individual_activations=True):
    main_writer = SummaryWriter(log_path)
    if log_individual_activations:
        activation_writers = [SummaryWriter(os.path.join(log_path, str(num)))
                              for num in range(num_stripes)]

    for epoch in range(num_epochs):
        if epoch == num_epochs - 1:
            net.num_active_stripes += relax_stripe_sparsity
        train_loss = train_epoch(net,
                                 criterion,
                                 optimizer,
                                 X_train,
                                 batch_size,
                                 batch_no)
        net.distort_prob = max(net.distort_prob - distort_prob_decay, 0)
        main_writer.add_scalar('train_loss', train_loss, epoch)
        if log_individual_activations:
            log_activation_data(net,
                                activation_writers,
                                X_test,
                                Y_test,
                                num_stripes,
                                epoch)


