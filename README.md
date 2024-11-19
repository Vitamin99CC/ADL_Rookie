# ADL_Gruppe_Rookie

## Repository

*programming language:* python 3.11<br>
*libraries:* argparse, os, time, einops, torch(torch 2.5.1, torchaudio 2.5.1, torchvision 0.20.1)
## Update and Log
### Update 17.11.2024
New function device_choose() replace code in run(), add option mps to suit the device in mac.<br>
Add transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip() into transform.Compose to complete the requirement a of task1.<br>
In the end of run(), Determine the current accuracy in epoch loop and save the model with the highest accuracy in folder models.<br>
Modify the test() to return the accuracy of tested model.
### Log 17.11.2024
ResNet18 without pre-trained weights can reach about 63% accuracy after 5 epochs.<br>
With pre-trained weights it can be 79% after 5 epochs.<br>

### Update 18.11.2024
Partially fix the class Attention(nn.Module) in my_models_skeleton to solve the task 1.2.<br>

### Update 19.11.2024
Fix the class Attention maybe, the accuracy after 5 epochs reach about 37%.<br>