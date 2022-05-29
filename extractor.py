import torch
from torch.autograd import Variable as V


class Extractor:
    def __init__(self, net, loss=None, eval_mode=False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.img = None
        self.mask = None
        self.nr = None
        if eval_mode:
            self.net.eval()
        else:
            self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=2e-4)
            self.loss = loss()
            self.old_lr = 2e-4

    def set_input(self, img_batch, mask_batch=None, nr_batch=None):
        self.img = V(img_batch.cuda())
        if mask_batch is not None:
            self.mask = V(mask_batch.cuda())
        if nr_batch is not None:
            self.nr = V(nr_batch.cuda())

    def optimize(self):
        self.optimizer.zero_grad()
        if self.nr is not None:
            pred, dist, dire = self.net.forward(self.img)
            loss = self.loss(self.mask, pred, self.nr, dist, dire)
        else:
            pred = self.net.forward(self.img)
            loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, nr_head=False):
        if nr_head:
            pred = self.net.forward(self.img)[0].squeeze().cpu().data.numpy()
        else:
            pred = self.net.forward(self.img).squeeze().cpu().data.numpy()
        return pred

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self):
        new_lr = self.old_lr * 0.2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.old_lr = new_lr
