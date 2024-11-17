# ADL_Gruppe_Rookie

## Repository

*programming language:* python 3.11<br>
*libraries:* argparse, os, torch(torch 2.5.1, torchaudio 2.5.1, torchvision 0.20.1)

### update 17.11.2024
New function device_choose() replace code in run(), add option mps to suit the device in mac.<br>
Add transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip() into transform.Compose to complete the requirement a of task1.<br>
In the end of run(), Determine the current accuracy in epoch loop and save the model with the highest accuracy in folder models.<br>
Modify the test() to return the accuracy of tested model.

