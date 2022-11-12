import csv
import os.path as osp 
import os


def plot(IMAGE_DIRECTORY, fname, x , y):
    # print(x)
    print(y)
    import matplotlib.pyplot as plt
    image_fname = osp.join(IMAGE_DIRECTORY,fname+".png")

    print(f"==> Saving plots at {image_fname}")
    if not os.path.isdir(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY)

    fig, ax = plt.subplots()

    #linewidth=2, markersize=12, line_style= "dashed"
    ax.scatter(x, y)
    
    ax.set_title("train loss vs num epochs")
    ax.set_xlabel("num_epochs")
    ax.set_ylabel("train loss")
    # ax.legend(loc= "lower right")
    # fig.suptitle()
    # fig.suptitle("\n".join(wrap(title, 70)), y=1.08)
    plt.savefig(image_fname)

log_dir_path = "try4/v4/20221109-19:21"
num_epochs= 50

with open(osp.join(log_dir_path, "logs.txt"), newline="") as f:
    reader = csv.reader(f)
    train_loss_list =[]
    for line in reader:
        if 'train loss' in line[0]:
            train_loss_list.append(float(line[0].split(" ")[5][5:]))

plot(log_dir_path, "train_loss_epochs", list(range(500)), train_loss_list)


