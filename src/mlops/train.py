from mlops.model import Model
from mlops.data import load_data
import torch
import hydra
import os
from hydra.utils import get_original_cwd
import wandb

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# @hydra.main(version_base = None, config_path = "configs", config_name = "config.yaml")
# def train(cfg):
def train():
    # batch_size = cfg.hyperparameters.batch_size
    # epochs = cfg.hyperparameters.epochs
    # lr = cfg.hyperparameters.lr
    # seed = cfg.hyperparameters.seed

    batch_size = 32
    epochs = 10
    lr = 1e-3
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Logging
    wandb.init(project='playing-cards-mlops',config={'epochs':epochs,'batch size':batch_size,
                                                     'learning rate':lr,'seed':seed})

    # Loading data
    train_set = load_data(split = "train")
    test_set = load_data(split = "test")
    model = Model().to(DEVICE)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    loss_fn =  torch.nn.CrossEntropyLoss(ignore_index=-1) # Using Cross entropy 
    rank_weight = 1
    suit_weight = 1 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    statistics = {"train_loss": [], "train_accuracy_suit": [], "train_accuracy_rank": []}

    # start training
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img = img.float() / 255.0  # convert to float in [0,1]
            img = img.to(DEVICE)
            print(target)
            target[:,0] = target[:,0].to(DEVICE) # 0 is rank
            target[:,1] = target[:,1].to(DEVICE) # 1 is suit

            #predict
            y_pred = model(img)
            
            # compute loss
            loss = suit_weight*loss_fn(y_pred['suit'], target[:,1]) + rank_weight*loss_fn(y_pred['rank'], target[:,0])  # calculating loss as sum of the seperate losses
            # gradient step
            loss.backward()
            optimizer.step()

            # Statistics
            statistics["train_loss"].append(loss.item())

            r_accuracy = (y_pred['rank'].argmax(dim=1) == target[:,0]).float().mean().item()
            s_accuracy = (y_pred['suit'].argmax(dim=1) == target[:,1]).float().mean().item()

            statistics["train_accuracy_rank"].append(r_accuracy)
            statistics["train_accuracy_suit"].append(s_accuracy)

            # Logging
            wandb.log({'loss':loss.item(),'rank accuracy':r_accuracy,'suit accuracy':s_accuracy})
            

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, "
                      f"rank_acc: {r_accuracy:.4f}, suit_acc: {s_accuracy:.4f}")

    print("Training complete")   
    save_dir = os.path.join(get_original_cwd(), "models")
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

if __name__ == "__main__":
    train()
