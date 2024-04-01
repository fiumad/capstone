import pandas as pd
import torch


class TransimpedanceData:
    def __init__(self):
        # self.raw_data = pd.read_csv("TransimpedanceData.csv")
        self.raw_gain_data = pd.read_csv("gain.csv")
        self.raw_power_data = pd.read_csv("power.csv")
        self.raw_bandwidth_data = pd.read_csv("bandwidth.csv")
        self.gain_data = []
        self.power_data = []
        self.bandwidth_data = []

        self.inputs = []
        self.targets = []

        # save the data from the csv file into a list of lists with the format [[frequency, wn, transimpedance]]
        for i in range(len(self.raw_gain_data)):
            for j in range(625):  # was 41
                col_title = self.raw_gain_data.columns[j + 1].split(" ")
                # print(col_title)
                # input("Press Enter to continue...")
                self.gain_data.append(
                    [
                        float(self.raw_gain_data.iloc[i, 0]),  # input width
                        # float(self.raw_gain_data.columns[j + 1][27:34]),  # tail width
                        float(col_title[4]),
                        float(
                            # self.raw_gain_data.columns[j + 1][38:]
                            col_title[2]
                        ),  # strip the column name to get the tail width
                        float(self.raw_gain_data.iloc[i, j + 1]),  # transimpedance
                    ]
                )
        # save the inputs and targets into separate lists
        for i in range(len(self.gain_data)):
            # Inputs: [Input Width, Load Resistance, Tail Width]
            self.inputs.append(
                [self.gain_data[i][0], self.gain_data[i][1], self.gain_data[i][2]]
            )

            # Targets: [Transimpedance]
            self.targets.append(self.gain_data[i][3])

        # ingest power data
        # TODO: Replace with data that sweeps input width
        # For now, use 2u
        for i in range(len(self.raw_power_data)):
            for j in range(25):
                col_title = self.raw_power_data.columns[j + 1].split(" ")
                # print(col_title)
                # input("Press Enter to continue...")

                self.power_data.append(
                    [
                        # [Input Width, Load Resistance, Tail Width, Power]
                        2.0e-6,
                        float(col_title[5]),
                        self.raw_power_data.iloc[i, 0],
                        float(self.raw_power_data.iloc[i, j + 1]),
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
                        float(
                            # self.raw_bandwidth_data.columns[j + 1][50:58]
                            col_title[5]
                        ),  # input width
                        float(
                            # self.raw_bandwidth_data.columns[j + 1][62:]
                            col_title[7]
                        ),  # Load Resistance
                        float(self.raw_bandwidth_data.iloc[i, 0]),  # tail width
                        float(self.raw_bandwidth_data.iloc[i, j + 1]),  # bandwidth
                    ]
                )

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
    if input("Print gain data? (y/n) ") == "y":
        for i in range(len(data.gain_data)):
            print(data.gain_data[i])

    if input("Print bandwidth data? (y/n) ") == "y":
        for i in range(len(data.bandwidth_data)):
            print(data.bandwidth_data[i])

    if input("Print power data? (y/n) ") == "y":
        for i in range(len(data.power_data)):
            print(data.power_data[i])
