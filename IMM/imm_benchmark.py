import os
import pickle as pkl

name = "livejournal1"
data_name = "../../data/dataset/"+name+"/"
model = "IC"
epsilon = 0.1

imm_results = {}
for k in range(1, 50):
    imm_results[k] = dict()
    comd = "./imm_discrete -dataset {} -k {} -model {} -epsilon {}".format(data_name, k, model, epsilon)
    res = os.popen(comd).readlines()
    imm_results[k]["res"] = res

    for s in res:
        if "g.seedSet=" in s:
            seed = s.split("=")[1].strip("\n").strip().split()
            seed = [int(i) for i in seed]
            imm_results[k]["seeds"] = seed
        if "Total Time" in s:
            t = float(s.split()[1])
            imm_results[k]["time"] = t
            print("Seed size = {}, Time = {}s".format(k, t))
    with open("../../data/imm_benchmark/"+name+".pkl", "wb") as f:
        pkl.dump(imm_results,f)    

