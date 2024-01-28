import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from argparse import ArgumentParser

from .model import ActionClassifier
from .dataset import FrameDataset


parser = ArgumentParser()
parser.add_argument("--save_path", type=str, required=False)
parser.add_argument("--load_path", type=str, required=False)
parser.add_argument("--rgb_dataset", type=str, required=True)
parser.add_argument("--of_dataset", type=str, required=True)
parser.add_argument("--train_split", type=str, required=True)
parser.add_argument("--test_split", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=False)
parser.add_argument("--epochs", type=int, required=True)

args = parser.parse_args()

def compute_class_weights(labels, device=torch.device("cuda")):
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    return class_weights

def main():
  batch_size = 12
  if args.batch_size is not None:
    batch_size = args.batch_size

  device = torch.device("cuda")

  if args.save_path is None:
    save_path = "./checkpoints/action_e"
  else: 
    save_path = args.save_path + "/action_e"

  dataset = FrameDataset(args.rgb_dataset, args.of_dataset, args.train_split)
  train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

  l1, l2, l3 = [], [], []
  for noun, verb, action in dataset.labels:
    l1.append(noun)
    l2.append(verb)
    l3.append(action)

  noun_weights = compute_class_weights(l1)
  verb_weights = compute_class_weights(l2)
  action_weights = compute_class_weights(l3)
  noun_criterion = nn.CrossEntropyLoss(weight=noun_weights)
  verb_criterion = nn.CrossEntropyLoss(weight=verb_weights)
  action_criterion = nn.CrossEntropyLoss(weight=action_weights)

  test_dataset = FrameDataset(args.rgb_dataset, args.of_dataset, args.test_split)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

  model = ActionClassifier().to(device)

  for p in model.parameters():
    p.requires_grad = True

  loss_weight = 0.5
  loss_weight_aux = 0.4

  optimizer = torch.optim.SGD([params for params in model.parameters() if params.requires_grad],lr = 0.01, momentum = 0.9)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98, last_epoch=-1, verbose = True)

  if args.load_path is not None:
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

  for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    noun_accuracy = 0
    verb_accuracy = 0
    action1_accuracy = 0
    action2_accuracy = 0
    for frames, of, labels in tqdm(train_dataloader):
        frames = frames.to(device)
        of = of.to(device)
        noun_labels, verb_labels, action_labels = labels
        noun_labels = noun_labels.to(device)
        verb_labels = verb_labels.to(device)
        action_labels = action_labels.to(device)

        optimizer.zero_grad()

        action_logits, action_logits_aux, noun_logits, verb_logits = model(frames, of)
        noun_loss = noun_criterion(noun_logits, noun_labels)
        verb_loss = verb_criterion(verb_logits, verb_labels)
        action_loss = action_criterion(action_logits, action_labels)
        action_loss_aux = action_criterion(action_logits_aux, action_labels)

        loss = action_loss + loss_weight * (noun_loss + verb_loss) + loss_weight_aux * action_loss_aux

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        noun_preds = torch.argmax(F.softmax(noun_logits, dim=1), dim=1, keepdim=False)
        verb_preds = torch.argmax(F.softmax(verb_logits, dim=1), dim=1, keepdim=False)
        action_preds = torch.argmax(F.softmax(action_logits, dim=1), dim=1, keepdim=False)

        noun_accuracy += (noun_preds == noun_labels).sum().item()
        verb_accuracy += (verb_preds == verb_labels).sum().item()
        action1_accuracy += ((noun_preds == noun_labels) & (verb_preds == verb_labels)).sum().item()
        action2_accuracy += (action_preds == action_labels).sum().item()

    lr_scheduler.step()

    total_loss /= len(train_dataloader)
    noun_accuracy /= len(dataset)
    verb_accuracy /= len(dataset)
    action1_accuracy /= len(dataset)
    action2_accuracy /= len(dataset)

    print()
    print(f"-------Epoch {epoch}--------------")
    print("Model loss: ", total_loss)
    print("Train noun accuracy: ", noun_accuracy)
    print("Train verb accuracy: ", verb_accuracy)
    print("Train action1 accuracy: ", action1_accuracy)
    print("Train action2 accuracy: ", action2_accuracy)


    model.eval()
    test_noun_accuracy = 0
    test_verb_accuracy = 0
    test_action1_accuracy = 0
    test_action2_accuracy = 0
    with torch.no_grad():
        for frames, of, labels in tqdm(test_dataloader):
            frames = frames.to(device)
            of = of.to(device)
            noun_labels, verb_labels, action_labels = labels
            noun_labels = noun_labels.to(device)
            verb_labels = verb_labels.to(device)
            action_labels = action_labels.to(device)

            action_logits, noun_logits, verb_logits = model(frames, of)
            noun_preds = torch.argmax(F.softmax(noun_logits, dim=1), dim=1, keepdim=False)
            verb_preds = torch.argmax(F.softmax(verb_logits, dim=1), dim=1, keepdim=False)
            action_preds = torch.argmax(F.softmax(action_logits, dim=1), dim=1, keepdim=False)

            test_noun_accuracy += (noun_preds == noun_labels).sum().item()
            test_verb_accuracy += (verb_preds == verb_labels).sum().item()
            test_action1_accuracy += ((noun_preds == noun_labels) & (verb_preds == verb_labels)).sum().item()
            test_action2_accuracy += (action_preds == action_labels).sum().item()

    test_noun_accuracy /= len(test_dataset)
    test_verb_accuracy /= len(test_dataset)
    test_action1_accuracy /= len(test_dataset)
    test_action2_accuracy /= len(test_dataset)

    print("Test noun accuracy: ", test_noun_accuracy)
    print("Test verb accuracy: ", test_verb_accuracy)
    print("Test action1 accuracy: ", test_action1_accuracy)
    print("Test action2 accuracy: ", test_action2_accuracy)

    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'train_loss': total_loss,
        'train_noun_accuracy': noun_accuracy,
        'train_verb_accuracy': verb_accuracy,
        'train_action1_accuracy': action1_accuracy,
        'train_action2_accuracy': action2_accuracy,
        'test_noun_accuracy': test_noun_accuracy,
        'test_verb_accuracy': test_verb_accuracy,
        'test_action1_accuracy': test_action1_accuracy,
        'test_action2_accuracy': test_action2_accuracy,
        }, save_path+str(epoch)+".pt")

 
if __name__ == "__main__":
   main()
