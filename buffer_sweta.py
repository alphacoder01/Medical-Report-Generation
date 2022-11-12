import torch.nn as nn
import torch

# class model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.a= nn.Linear(5,6)
#         self.b= nn.Linear(6,5)
#         self.relu = nn.ReLU()

#     def forward(self,x):
#         out = self.a(x)
#         out = self.relu(out)
#         print("out: ",out.shape)
#         out = self.b(x)
#         out = self.relu(out) 
#         out= nn.Linear(5,5)(out)
#         out = self.a(out)

# x = torch.rand((1,5))
# print(x.shape)


# model1 = model()
# print(model1.a.weight.shape)
# modules= list(model1.children())
# model2= nn.Sequential(*modules)

# # print(model())
# # print("\n")
# # print(modules)
# assert model1(x)== model2(x)


from torchvision.models import resnet18


model1= resnet18()
modules = list(model1.children())[:-1]

modules.append(nn.Flatten())
modules.append(nn.Linear(512,1000))

model2 = nn.Sequential(*modules)


x = torch.rand((1, 3, 256,256))
print(model2(x).shape)
assert(model1(x)==model2(x)).all()


