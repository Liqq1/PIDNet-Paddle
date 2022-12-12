import paddle
import paddle.nn as nn
import paddle.nn.functional as F

BatchNorm2d = nn.BatchNorm2D
bn_mom = 0.9
algc = False

#主要修改内容：BatchNorm2D, Conv2D(bias_attr), ReLU(), sigmoid, concat,AvgPool2D
#ReLU()反向传播？

class BasicBlock(nn.Layer):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=3, stride=stride,
                                padding=1, bias_attr=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3,
                                padding=1, bias_attr=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Layer):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, kernel_size=1,
                               bias_attr=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class segmenthead(nn.Layer):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2D(inplanes, interplanes, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(interplanes, outplanes, kernel_size=1, padding=0)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out


class DAPPM(nn.Layer):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2D):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2D(kernel_size=5, stride=2, padding=2,exclusive=False),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2D(kernel_size=9, stride=4, padding=4,exclusive=False),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2D(kernel_size=17, stride=8, padding=8,exclusive=False),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2D((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(branch_planes * 5, outplanes, kernel_size=1, bias_attr=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, outplanes, kernel_size=1, bias_attr=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[3])))
       
        out = self.compression(paddle.concat(x_list, 1)) + self.shortcut(x)
        return out 

class PAPPM(nn.Layer):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2D):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2D(kernel_size=5, stride=2, padding=2,exclusive=False),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2D(kernel_size=9, stride=4, padding=4,exclusive=False),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2D(kernel_size=17, stride=8, padding=8,exclusive=False),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2D((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1,  bias_attr=False),
                                    )

        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1,  bias_attr=False),
                                    )
        
        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*4, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(branch_planes*4, branch_planes*4, kernel_size=3, padding=1, groups=4,  bias_attr=False),
                                    )

      
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(branch_planes * 5, outplanes, kernel_size=1,  bias_attr=False),
                                    )
        
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, outplanes, kernel_size=1,  bias_attr=False),
                                    )


    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        
        scale_out = self.scale_process(paddle.concat(scale_list, 1))
       
        out = self.compression(paddle.concat([x_,scale_out], 1)) + self.shortcut(x)
        return out



class PagFM(nn.Layer):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2D):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
                                nn.Conv2D(in_channels, mid_channels, kernel_size=1, bias_attr=False),
                                BatchNorm(mid_channels)
                                )
        self.f_y = nn.Sequential(
                                nn.Conv2D(in_channels, mid_channels, kernel_size=1, bias_attr=False),
                                BatchNorm(mid_channels)
                                )
        if with_channel:
            self.up = nn.Sequential(
                                    nn.Conv2D(mid_channels, in_channels, kernel_size=1, bias_attr=False),
                                    BatchNorm(in_channels)
                                   )
        if after_relu:
            self.relu = nn.ReLU()
        
    def forward(self, x, y):
        input_size = x.shape
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        
        if self.with_channel:
            sim_map = F.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = F.sigmoid(paddle.sum(x_k * y_q, axis=1).unsqueeze(1))
        
        y = F.interpolate(y, size=[input_size[2], input_size[3]],mode='bilinear', align_corners=False)
        x = (1-sim_map)*x + sim_map*y
        return x
    
class Light_Bag(nn.Layer):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2D):
        super(Light_Bag, self).__init__()
        self.conv_p = nn.Sequential(
                                nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False),
                                BatchNorm(out_channels)
                                )
        self.conv_i = nn.Sequential(
                                nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False),
                                BatchNorm(out_channels)
                                )
        
    def forward(self, p, i, d):
        edge_att = F.sigmoid(d)
        
        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(i + edge_att*p)
        
        return p_add + i_add
    

class DDFMv2(nn.Layer):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2D):
        super(DDFMv2, self).__init__()
        self.conv_p = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(),
                                nn.Conv2D(in_channels, out_channels, 
                                          kernel_size=1, bias_attr=False),
                                BatchNorm(out_channels)
                                )
        self.conv_i = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(),
                                nn.Conv2D(in_channels, out_channels, 
                                          kernel_size=1, bias_attr=False),
                                BatchNorm(out_channels)
                                )
        
    def forward(self, p, i, d):
        edge_att = F.sigmoid(d)
        
        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(i + edge_att*p)
        
        return p_add + i_add

class Bag(nn.Layer):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2D):
        super(Bag, self).__init__()

        self.conv = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(),
                                nn.Conv2D(in_channels, out_channels, 
                                          kernel_size=3, padding=1, bias_attr=False)                  
                                )

        
    def forward(self, p, i, d):
        edge_att = F.sigmoid(d)
        return self.conv(edge_att*p + (1-edge_att)*i)
    


if __name__ == '__main__':

    x = paddle.rand(4, 64, 32, 64).cuda()
    y = paddle.rand(4, 64, 32, 64).cuda()
    z = paddle.rand(4, 64, 32, 64).cuda()
    net = PagFM(64, 16, with_channel=True).cuda()
    
    out = net(x,y)