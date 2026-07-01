import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data_and_plot():
    # 读取csv，跳过前6行无用表头，以第6行作为列名
    df = pd.read_csv(r"C:\Users\wngan\Desktop\采集卡\r3000A1A2 2026_07_01 11_43_01.csv"
                     , skiprows=5, encoding="gbk")

    # 重命名列简化使用
    df.columns = ["Time", "A1", "A2"]

    # 查看前10行数据
    print(df.head(10))

    # 提取时间、双通道位移信号
    t = df["Time"].values
    sig_a1 = df["A1"].values
    sig_a2 = df["A2"].values

    sig_a1_change, sig_a2_change = data_change(sig_a1, sig_a2)
    mask = (t > 80) & (t < 81)
    t_slice = t[mask]
    sig_a1_change_slice = sig_a1_change[mask]
    sig_a2_change_slice = sig_a2_change[mask]

    # 简单绘图查看波形
    plt.figure(figsize=(12,5))
    plt.plot(t_slice, sig_a1_change_slice, label="Channel A1 (mm)")
    plt.plot(t_slice, sig_a2_change_slice, label="Channel A2 (mm)")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (mm)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 第二张新画布：绘制 A1 - 2*A2
    combined_sig = sig_a1_change_slice - 2 * sig_a2_change_slice
    plt.figure(figsize=(12, 5))
    plt.plot(t_slice, combined_sig, color="red", label="Signal = A1 - 2*A2")
    plt.xlabel("Time (s)")
    plt.ylabel("Combined Displacement (mm)")
    plt.title("Combined Signal A1 - 2*A2")
    plt.legend()
    plt.grid(True)
    plt.show()

def data_change(sig_a1: np.ndarray, sig_a2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sig_a1 = sig_a1 - sig_a1[0]
    sig_a2 = sig_a2 - sig_a2[0]
    sig_a2 = -sig_a2
    return sig_a1, sig_a2



if __name__ == '__main__':
    read_data_and_plot()