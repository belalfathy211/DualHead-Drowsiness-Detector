import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from FolderDataset import FolderDataset
from model import TwoHeadModel

EPOCHS = 15
LEARNING_RATE = 0.001
img_size = 112
train_eye_path = "/home/belal/projects/TwoHead/data/eye/Train"
val_eye_path = "/home/belal/projects/TwoHead/data/eye/Val"
train_mouth_path = "/home/belal/projects/TwoHead/data/Yawn/Train"
val_mouth_path = "/home/belal/projects/TwoHead/data/Yawn/Val"

train_transform = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.RandomVerticalFlip(.5),
    transforms.RandomHorizontalFlip(.5),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor()
])

eye_train_dataset = FolderDataset(root=train_eye_path, transform=train_transform)
eye_train_dataloader = DataLoader(eye_train_dataset,batch_size=32,shuffle=True,num_workers=8)
eye_val_dataset = FolderDataset(root=val_eye_path, transform=val_transform)
eye_val_dataloader = DataLoader(eye_val_dataset,batch_size=32,num_workers=8)

mouth_train_dataset = FolderDataset(root=train_mouth_path, transform=train_transform)
mouth_train_dataloader = DataLoader(mouth_train_dataset,batch_size=32,shuffle=True,num_workers=8)
mouth_val_dataset = FolderDataset(root=val_mouth_path, transform=val_transform)
mouth_val_dataloader = DataLoader(mouth_val_dataset,batch_size=32,num_workers=8)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TwoHeadModel().to(device)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    total_e = 0
    total_m = 0
    train_total_correct_eye = 0
    train_total_correct_mouth = 0
    counter = 0
    for (Xe,ye),(Xm,ym) in zip(eye_train_dataloader, mouth_train_dataloader):
        Xe,ye,Xm,ym = Xe.to(device), ye.to(device), Xm.to(device), ym.to(device)
        counter += 1
        y_pred_eye = model(Xe)
        y_pred_mouth = model(Xm)
        loss1 = loss_fn(y_pred_eye[0], ye.unsqueeze(-1).float())
        loss2 = loss_fn(y_pred_mouth[1], ym.unsqueeze(-1).float())
        loss = loss1 + loss2
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_e += y_pred_eye[0].shape[0]
        total_m += y_pred_mouth[1].shape[0]
        train_total_correct_eye += ((y_pred_eye[0]>.5).int() == ye.unsqueeze(-1)).sum().item()
        train_total_correct_mouth += ((y_pred_mouth[1]>.5).int() == ym.unsqueeze(-1)).sum().item()
    print(f"Epoch: {epoch} :- \nTrain Loss: {train_loss/counter:.4f} | Eye Train Acc: {(train_total_correct_eye/total_e*100):.2f}% | Mouth Train Acc: {(train_total_correct_mouth/total_m*100):.2f}%")
    model.eval()
    with torch.inference_mode():
        val_loss = 0
        total_correct = 0
        for X, y in eye_val_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred[0], y.unsqueeze(-1).float())
            val_loss += loss.item()
            total_correct += ((y_pred[0] > .5).int() == y.unsqueeze(-1)).sum().item()
        print(f"Eye Val Loss: {val_loss / len(eye_val_dataloader):.4f} | Eye Val Acc: {(total_correct / len(eye_val_dataset) * 100):.2f}%")
        val_loss = 0
        total_correct = 0
        for X, y in mouth_val_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred[1], y.unsqueeze(-1).float())
            val_loss += loss.item()
            total_correct += ((y_pred[1] > .5).int() == y.unsqueeze(-1)).sum().item()
        print(f"Mouth Val Loss: {val_loss / len(mouth_val_dataloader):.4f} | Mouth Val Acc: {(total_correct / len(mouth_val_dataset) * 100):.2f}%")

        print("_"*30)
