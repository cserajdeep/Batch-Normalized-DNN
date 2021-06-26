# Deep Neural Network   
(1) BN_DNN with 4 layers and 1 output layer. The model gives 100% test accuracy for Iris 7:3 split. <br> 
(2) DNN_LITE with 2 layers and 1 output layer. The model provides 91.11% test accuracy for same 7:3 split.

<table style="width:50%">
  <tr>
    <th>Model</th>
    <th>Accuracy (%)</th>
    <th>AUC</th> 
    <th>#Param</>
  </tr>
  <tr>
    <td>BN_DNN</td>
    <td>100</td>
    <td>1.00</td>
    <td>202,755</td>
  </tr>
  <tr>
    <td>DNN_LITE</td>
    <td>91.11</td>
    <td>0.978</td>
    <td>2,953</td>
  </tr>
</table>

<p><strong>Updated:</strong> 26-June-2021.</p>


# Batch Normalized Neural Architecture (BN_DNN)
```ruby
class BN_DNN(nn.Module):
    """Feedfoward neural network with 4 hidden layer"""
    def __init__(self, in_size, out_size):
        super().__init__()
        # hidden layer 1
        self.linear1 = nn.Linear(in_size, 256)
        nn.BatchNorm1d(256)    #applying batch norm
        # hidden layer 2
        self.linear2 = nn.Linear(256, 512)
        nn.BatchNorm1d(512)    #applying batch norm
        # hidden layer 3
        self.linear3 = nn.Linear(512, 128)
        nn.BatchNorm1d(128)    #applying batch norm
        # hidden layer 4
        self.linear4 = nn.Linear(128, 32)
        nn.BatchNorm1d(32)    #applying batch norm
        # output layer
        self.linear5 = nn.Linear(32, out_size)
```
# Batch Normalized Neural Architecture (DNN_LITE)
```ruby
class DNN_LITE(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(DNN_LITE, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        nn.BatchNorm1d(50)
        self.layer2 = nn.Linear(50, 50)
        nn.BatchNorm1d(50)
        self.layer3 = nn.Linear(50, out_dim)
```

# Batch Normalized DNN
