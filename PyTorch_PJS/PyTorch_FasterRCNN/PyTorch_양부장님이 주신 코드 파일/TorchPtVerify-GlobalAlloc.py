import torch
import torchvision

loaded = torch.jit.load(r"d:\fasterrcnn_best_new2.pt")

dummy = torch.zeros(
    (3,2048,4096),
    dtype=torch.float32,
    device="cuda")

loaded.cuda()
loaded.eval()

with torch.no_grad():

    for i in range(100):

        boxes, scores, labels = loaded(dummy)

        if i % 10 == 0:
            print(i)
