import torch
import paddle

path="/home/aistudio/PIDNet_M_ImageNet.pth.tar"
state_dict=torch.load(path)

new_state_dict={}
for k,v in state_dict.items():
    new_k=k.replace("model.","")
    if 'criterion' in k:
        continue
    if 'seghead' in k:
        continue
    if 'num_batches_tracked' in k:
        continue
    if 'mean' in new_k:
        new_k = new_k.replace( "running_mean","_mean")
    if "var" in new_k:
        new_k = new_k.replace( "running_var","_variance")
    new_state_dict[new_k]=v.cpu().numpy()
    
# print(new_state_dict['conv1.0.weight'])

paddle.save(new_state_dict, '/home/aistudio/PIDNet_M_val.pdparams')

# path2="PIDNet_M_val.pdparams"
# new_state_dict2=paddle.load(path2)
# print(new_state_dict2['conv1.0.weight'])



