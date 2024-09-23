import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.null = nn.Linear(1, 1)

    def forward(self, x):
        return self.null(x)


class Loss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.null = nn.Linear(1, 1)

    def forward(self, x):
        return self.null(x)


model = Model()
loss = Loss()

optim_model = optim.AdamW(model.parameters(), lr=5e-4)
optim_loss = optim.AdamW(loss.parameters(), lr=1e-5)

sch_model = optim.lr_scheduler.StepLR(optim_model, 30, 0.5)
sch_loss = optim.lr_scheduler.StepLR(optim_loss, 45, 0.8)

model_lr = []
loss_lr = []
for _ in range(300):
    m_lr = optim_model.param_groups[0]['lr']
    l_lr = optim_loss.param_groups[0]['lr']
    sch_model.step()
    sch_loss.step()
    model_lr.append(m_lr)
    loss_lr.append(l_lr)

plt.figure(figsize=(9, 6))
plt.title('StepLR Scheduler', fontsize=25)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('lr', fontsize=20)
plt.yscale('log')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.plot(list(range(300)), model_lr, linewidth=3, label='model parameters')
plt.plot(list(range(300)), loss_lr, linewidth=3, label='learnable loss weights')
plt.legend(fontsize=20)
plt.savefig('lr.svg')
