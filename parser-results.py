import glob
import numpy as np

KEYS = ["Model", "Loss", "Accuracy", "AUC", "Precision", "Recall", "F1 score", "Time (secs)"]

def list_files(prefix):
    return glob.glob(f"./{prefix}*.txt")

def read_file(foo):
    f = open(foo, "r")
    return f.readlines()

def parse_file(data):
    result = dict()
    clean_data = lambda s: s.replace(" ", "").replace("\n", "")
    last_id = None

    for line in data:
        if (len(line.strip()) <= 0):
            continue

        words = clean_data(line).split(":")
        if words[0] == KEYS[0]:
            result[words[1]] = dict()
            last_id = words[1]
        else:
            result[last_id][words[0]] = words[1]

    return result

def get_metrics(datalist):
    metrics = dict()
    
    # accomodate the data to obtaine the metrics
    for data in datalist:
        for item in data.items():
            key = item[0]
            if key not in metrics.keys():
                metrics[key] = dict()
            else:
                for mk in item[1].keys():
                    if mk not in metrics[key]:
                        metrics[key][mk] = []
                    metrics[key][mk].append(float(item[1][mk]))

    # calculate the metrics
    avg_metrics = dict()
    for item in metrics.items():
        for metric in item[1].keys():
            if item[0] not in avg_metrics.keys():
                avg_metrics[item[0]] = dict()

            mean, std = np.mean(item[1][metric]), np.std(item[1][metric])
            if metric != KEYS[0]:
                avg_metrics[item[0]][metric] = "{:.3f}".format(mean) + " ** " + "{:.2f}".format(std)
            else:
                avg_metrics[item[0]][metric] = f"{mean} +- {std}"

    return avg_metrics

if __name__ == "__main__":
    prefixes = ["diabetes"] # "cancer", "diabetes", "malaria"

    data_list = []
    for prefix in prefixes:
        files = list_files(prefix)
        for file in files:
            raw_data = read_file(file)
            data = parse_file(raw_data)
            data_list.append(data)
   
    metrics = get_metrics(data_list)
    for key in metrics.keys():
        print(key)
        for m in metrics[key].keys():
            print(m, metrics[key][m])
        print()

