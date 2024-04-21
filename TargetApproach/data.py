import pandas as pd
import torch


class CircuitData:
    def __init__(self):
        self.raw_gain_data = pd.read_csv("gain.csv")
        self.raw_power_data = pd.read_csv("power.csv")
        self.raw_bandwidth_data = pd.read_csv("bandwidth.csv")
        self.gain_data = []
        self.power_data = []
        self.bandwidth_data = []
        self.data = []
        self.inputs = []
        self.targets = []

        # save the data from the csv file into a list of lists with the format [[wn, load resistance, tail width, transimpedance]]
        for i in range(len(self.raw_gain_data)):
            for j in range(625):  # was 41
                col_title = self.raw_gain_data.columns[j + 1].split(" ")
                # print(col_title)
                # input("Press Enter to continue...")
                self.gain_data.append(
                    [
                        float(col_title[6]),  # input width
                        float(col_title[8]),  # load resistance
                        float(self.raw_gain_data.iloc[i, 0]),  # tail width
                        float(self.raw_gain_data.iloc[i, j + 1]),  # transimpedance
                    ]
                )
        # ingest power data
        for i in range(len(self.raw_power_data)):
            for j in range(625):
                col_title = self.raw_power_data.columns[j + 1].split(" ")
                # print(col_title)
                # input("Press Enter to continue...")

                self.power_data.append(
                    [
                        # [Input Width, Load Resistance, Tail Width, Power]
                        float(col_title[5]),  # wn
                        float(col_title[7]),  # load resistance
                        float(self.raw_power_data.iloc[i, 0]),  # tail width
                        float(self.raw_power_data.iloc[i, j + 1]),  # power
                    ]
                )

        # ingest bandwidth data
        for i in range(len(self.raw_bandwidth_data)):
            # Inputs: [Input Width, Load Resistance, Tail Width]
            for j in range(625):
                col_title = self.raw_bandwidth_data.columns[j + 1].split(" ")
                # print(col_title)
                # input("Press Enter to continue...")
                self.bandwidth_data.append(
                    [
                        float(col_title[5]),  # input width
                        float(col_title[7]),  # Load Resistance
                        float(self.raw_bandwidth_data.iloc[i, 0]),  # tail width
                        float(self.raw_bandwidth_data.iloc[i, j + 1]),  # bandwidth
                    ]
                )

        # zip the data together such that the tensors to be fed into the model
        # are in the format [input width, load resistance, tail width, gain, bandwidth, power]
        print(len(self.gain_data), len(self.bandwidth_data), len(self.power_data))
        for i in range(len(self.gain_data)):
            tempi = []  # circuit parameters
            temp = []  # circuit performance

            tempi.append(self.gain_data[i][0] * 1000000)  # input width
            tempi.append(self.gain_data[i][1] / 10000)  # load resistance
            tempi.append(self.gain_data[i][2] * 1000000)  # tail width

            temp.append(self.gain_data[i][3] / 10000)  # gain normalized
            temp.append(self.bandwidth_data[i][3] / 10000000000)  # bandwidth normalized
            temp.append(self.power_data[i][3] * 100)  # power normalized

            self.data.append([temp, tempi])

        # split the data into inputs and targets
        for i in range(len(self.data)):
            self.inputs.append(self.data[i][0])
            self.targets.append(self.data[i][1])

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
    data = CircuitData()
    if input("Print cleaned gain data? (y/n) ") == "y":
        for i in range(len(data.gain_data)):
            print(data.gain_data[i])

    if input("Print cleaned bandwidth data? (y/n) ") == "y":
        for i in range(len(data.bandwidth_data)):
            print(data.bandwidth_data[i])

    if input("Print cleaned power data? (y/n) ") == "y":
        for i in range(len(data.power_data)):
            print(data.power_data[i])

    if input("Print all cleaned data? (y/n) ") == "y":
        for i in range(len(data.data)):
            print(data.data[i])
