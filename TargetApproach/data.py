import pandas as pd
import torch


class TransimpedanceData:
    def __init__(self):
        # self.raw_data = pd.read_csv("TransimpedanceData.csv")
        self.raw_data = pd.read_csv("TransimpedanceDataV2.csv")
        self.data = []
        self.inputs = []
        self.targets = []
        # save the data from the csv file into a list of lists with the format [[frequency, wn, transimpedance]]
        for i in range(len(self.raw_data)):
            for j in range(321):  # was 41
                self.data.append(
                    [
                        float(self.raw_data.iloc[i, 0]),
                        float(self.raw_data.columns[j + 1][28:]),  # added [28:]
                        float(self.raw_data.iloc[i, j + 1]),
                    ]
                )
        # save the inputs and targets into separate lists
        for i in range(len(self.data)):
            self.inputs.append([self.data[i][0], self.data[i][1]])
            self.targets.append(self.data[i][2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float),
        }

    def print_data(self, idx):
        print(
            torch.tensor(self.inputs[idx], dtype=torch.float),
            torch.tensor(self.targets[idx], dtype=torch.float),
        )


if __name__ == "__main__":
    data = TransimpedanceData()
    print(data.data)
