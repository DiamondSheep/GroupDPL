import torch
import torch.nn as nn

class demo(nn.Module):
    def __init__(self, in_channels, out_channels, dict_size):
        super().__init__()
        self.dict_conv = nn.Conv2d(in_channels=in_channels, out_channels=dict_size, 
                                kernel_size=3, stride=1, padding=0)
        self.coef_conv = nn.Conv2d(in_channels=dict_size, out_channels=out_channels,
                                kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        out = self.dict_conv(x)
        print(out.shape)
        out = self.coef_conv(x)
        return out 

if __name__ == "__main__":
    dict_size = 2
    in_map = torch.rand([1, 3, 3, 3])
    model = demo(3, 2, dict_size)
    out_map = model(in_map)
    print(out_map.shape)

exit(0)

channel_in = 3
channel_out = 6
dict_size = 2
dic_layer = torch.rand([dict_size, channel_in, 3, 3])
coef_layer = torch.rand([channel_out, dict_size, 1, 1])

dic_layer_expand = dic_layer.view(dict_size, channel_in * 3 * 3).t()
coef_layer_expand = coef_layer.view(channel_out, dict_size).t()

approx_layer = torch.mm(dic_layer_expand, coef_layer_expand)
approx_layer = approx_layer.view(channel_out, channel_in, 3, 3)

in_map = torch.rand([3, 3, channel_in])
out_map = []

for c in range(channel_out):
    p = 0.0
    for i in range(channel_in):
        p += torch.sum(torch.sum(in_map[:, :, i].mul(approx_layer[c, i, :, :]), dim=0))
    out_map.append(p)
print(out_map)

out_map = []
for c in range(dict_size):
    p = 0.0
    for i in range(channel_in):
        p += torch.sum(torch.sum(in_map[:, :, i] * dic_layer[c, i, :, :]))
    out_map.append(p)
out_map_2 = []
for c in range(channel_out):
    p = 0.0
    for i in range(dict_size):
        p += torch.sum(torch.sum(out_map[i].mul(coef_layer[c, i, :, :])))
    out_map_2.append(p)    
print(out_map_2)