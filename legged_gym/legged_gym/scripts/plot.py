
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
sns.set_context('poster')
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import json 

step = ["90-100","100-110","110-120","120-130","130-140","140-150"]*2

# noarm = [0.5935,
# 0.58925,
# 0.55525,
# 0.49175,
# 0.35425,
# 0.1775]

# noarm = [0.6686016441,
# 0.66204021,
# 0.6396163179,
# 0.593760611,
# 0.468242192,
# 0.3004016767]

# noarm = [
# 0.05738093635,
# 0.06065122092,
# 0.06843607701,
# 0.08626548698,
# 0.1220455327,
# 0.1440858775
# ]

noarm=[
0.009396479114,
0.01112649478,
0.01634628222,
0.02930154579,
0.04894379224,
0.05486640917
]
# ours = [
# 0.9005,
# 0.847,
# 0.754,
# 0.6865,
# 0.56175,
# 0.3935
# ]

# ours = [0.9280770877,
# 0.8813230255,
# 0.7813919502,
# 0.7192104669,
# 0.6365549795,
# 0.5176440015
# ]
# ours=[
# 0.1003354476,
# 0.09871745135,
# 0.09399175123,
# 0.09647944507,
# 0.103662167,
# 0.1138054601
# ]
ours=[
0.01507993571,
0.01496752015,
0.01474456034,
0.02044394239,
0.03403802503,
0.05048270882
]

if __name__ == "__main__":
    policy = ["Ours"]*len(ours) + ["Frozen"]*len(noarm)
    # step = np.concatenate([np.arange(len(OURS)),np.arange(len(DECOUPLE)),np.arange(len(NOARM)),np.arange(len(PPO))])

    data = np.concatenate([ours,noarm])
    df = pd.DataFrame()
    df["Policy"] = policy
    df["Step"] = step
    df["data"] = data   
    print(data)
    fig = plt.figure() 
    # ax = sns.histplot(data=df)
    ax = sns.barplot(x="Step",y="data", hue="Policy", data=df)
    fig.set_size_inches(10, 8.)
    plt.legend(bbox_to_anchor=(-0.01, 0.95, 1.0, 0.01),ncol=2,fontsize=25)
    plt.xlabel("External Perturbation", fontsize=25)
    plt.ylabel("Linear Velocity Tracking Error $\downarrow$", fontsize=25)
    plt.savefig("lv.png", bbox_inches="tight", pad_inches=0.0, dpi=300)
    plt.show()
    
    
# step = np.arange(2000)

# with open("/home/ravenhuang/amp/locomani/legged_gym/logs/hist_res/decouple_cmd_[0.0, 0.0]_it_1002_20230204-153026.json", "r") as fp:
#     decouple = json.load(fp)
                
# DECOUPLE_a = np.array(decouple["len_buffer"])
# DECOUPLE = DECOUPLE_a[np.where(DECOUPLE_a<=1001)]

# with open("/home/ravenhuang/amp/locomani/legged_gym/logs/hist_res/dog_only_cmd_[0.0, 0.0]_it_1002_20230204-170055.json", "r") as fp:
#     noarm = json.load(fp)
                
# NOARM_a = np.array(noarm["len_buffer"])
# NOARM = NOARM_a[np.where(NOARM_a<=1001)]


# with open("/home/ravenhuang/amp/locomani/legged_gym/logs/hist_res/ours_cmd_[0.0, 0.0]_it_1002_20230204-165801.json", "r") as fp:
#     ours = json.load(fp)
                
# OURS_a = np.array(ours["len_buffer"])
# OURS = OURS_a[np.where(OURS_a<=1001)]


# with open("/home/ravenhuang/amp/locomani/legged_gym/logs/hist_res/ppo_scratch_cmd_[0.0, 0.0]_it_1002_20230204-165454.json", "r") as fp:
#     ppo = json.load(fp)
                
# PPO_a = np.array(ppo["len_buffer"])
# PPO = PPO_a[np.where(PPO_a<=1001)]



# if __name__ == "__main__":
#     policy = ["Ours"]*len(OURS) + ["Decouple"]*len(DECOUPLE) + ["Frozen"]*len(NOARM) + ['PPO']*len(PPO)


#     data = np.concatenate([OURS,DECOUPLE,NOARM,PPO])
#     df = pd.DataFrame()
#     df["Policy"] = policy
#     df["data"] = data   
#     print(data)
#     fig = plt.figure() 
#     ax = sns.histplot(x="data", hue="Policy", data=df, bins=30, kde=True,log_scale=(True,True))
#     fig.set_size_inches(10.5, 7.)
#     plt.legend(bbox_to_anchor=(-0.01, 0.95, 1.0, 0.01),ncol=2,fontsize=25)
#     plt.xlabel("Episode Length", fontsize=30)
#     plt.ylabel("Number of robot", fontsize=25)
#     plt.savefig("runtime.png", bbox_inches="tight", pad_inches=0.0, dpi=300)
#     plt.show()
