
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ROTATE_SPEED_RPM = 3180  # 转速
REMARK = "N"
WRITE_TO_CSV = True  # 是否写入csv文件
TIME_DOMAIN = False  # 时域信号/频域分析
TIME_ORIGINAL = False  # 原始信号
SLICE_DOWN = 262  # 数据截取区间下
SLICE_UP = 272  # 数据截取区间上

def read_data_and_plot():
    # 读取csv，跳过前6行无用表头，以第6行作为列名
    df = pd.read_csv(r"C:\Users\wngan\Desktop\采集卡\r3180LL 2026_07_03 10_28_40.csv"
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
    mask = (t >= SLICE_DOWN) & (t <= SLICE_UP)
    t_slice = t[mask]
    sig_a1_change_slice = sig_a1_change[mask]
    sig_a2_change_slice = sig_a2_change[mask]

    if TIME_ORIGINAL:
        plot_time_original(t, sig_a1, sig_a2)
    elif TIME_DOMAIN:
        plot_time_domain(t_slice, sig_a1_change_slice, sig_a2_change_slice)
    else:
        plot_fft_domain(t_slice, sig_a1_change_slice, sig_a2_change_slice)


def data_change(sig_a1, sig_a2):
    sig_a1 = sig_a1 - np.mean(sig_a1[:int(10/0.001)])
    sig_a2 = sig_a2 - np.mean(sig_a2[:int(10/0.001)])
    sig_a2 = - sig_a2
    return sig_a1, sig_a2

def fft_analysis(t, signal):
    """
    傅里叶变换计算频谱
    :param t: 时间数组
    :param signal: 时域信号
    :return: freq, amp 频率、幅值
    """
    N = len(signal)
    # 采样时间间隔
    dt = np.mean(np.diff(t))
    fs = 1 / dt  # 采样频率
    # FFT
    fft_vals = np.fft.fft(signal)
    # 单边频谱修正：直流不乘2，交流乘2
    amp_raw = np.abs(fft_vals[:N // 2]) / N
    freq = np.fft.fftfreq(N, d=dt)[:N // 2]

    amp = amp_raw.copy()
    amp[1:] = amp[1:] * 2  # 仅对f>0的交流分量×2，直流amp[0]保持原值
    dc_component = np.mean(signal)
    return freq, amp, fs, dc_component

def plot_time_original(x, y1, y2):
    # 简单绘图查看波形
    plt.figure(figsize=(12, 5))
    plt.plot(x, y1, label="Channel A1 (mm)")
    plt.plot(x, y2, label="Channel A2 (mm)")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (mm)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_time_domain(x, y1, y2):
    # 简单绘图查看波形
    plt.figure(figsize=(12, 5))
    plt.plot(x, y1, label="Channel A1 (mm)")
    plt.plot(x, y2, label="Channel A2 (mm)")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (mm)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 第二张新画布：绘制 A1 - 2*A2
    combined_sig = y1 - 2 * y2
    plt.figure(figsize=(12, 5))
    plt.plot(x, combined_sig, color="red", label="Signal = A1 - 2*A2")
    plt.xlabel("Time (s)")
    plt.ylabel("Combined Displacement (mm)")
    plt.title("Combined Signal A1 - 2*A2")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fft_domain(x, y1, y2):
    # FFT calculation for two channels, include DC component
    freq1, amp1, fs1, dc_a1 = fft_analysis(x, y1)
    freq2, amp2, fs2, dc_a2 = fft_analysis(x, y2)
    # 计算峰峰值
    pp_a1 = calc_peak_to_peak(y1)
    pp_a2 = calc_peak_to_peak(y2)
    # 二倍差值
    combined_sig = y1 - 2 * y2
    max_diff = np.max(np.abs(combined_sig))

    # 获取主频、主幅值
    if len(freq1) > 0:
        max_idx1 = np.argmax(amp1)
        main_freq_a1 = freq1[max_idx1]
        main_amp_a1 = amp1[max_idx1]
    else:
        main_freq_a1 = 0.0
        main_amp_a1 = 0.0

    if len(freq2) > 0:
        max_idx2 = np.argmax(amp2)
        main_freq_a2 = freq2[max_idx2]
        main_amp_a2 = amp2[max_idx2]
    else:
        main_freq_a2 = 0.0
        main_amp_a2 = 0.0

    # ========== 1. 控制台打印指标 ==========
    print(f"\nSampling frequency: fs = {fs1:.2f} Hz")
    print(f"==== DC Component ====")
    print(f"Channel A1 DC: {dc_a1:.6f}")
    print(f"Channel A2 DC: {dc_a2:.6f}")
    print(f"\n==== Main Frequency & Amplitude ====")
    print(f"A1 Main Freq: {main_freq_a1:.2f} Hz, Main Amp: {main_amp_a1:.4f}")
    print(f"A2 Main Freq: {main_freq_a2:.2f} Hz, Main Amp: {main_amp_a2:.4f}")
    print(f"\n==== Peak-to-Peak Value ====")
    print(f"A1 Peak-Peak: {pp_a1:.6f}")
    print(f"A2 Peak-Peak: {pp_a2:.6f}")
    print(f"\n==== Max Diff ====")
    print(f"| A1 - 2 * A2 | = {max_diff}")

    # ========== 2. 构造指标DataFrame，写入CSV ==========
    result_data = [
        {
            "Rotation_Speed": ROTATE_SPEED_RPM,
            "Channel": "A1",
            "DC_Component": round(dc_a1, 6),
            "Main_Frequency_Hz": round(main_freq_a1, 2),
            "Main_Amplitude": round(main_amp_a1, 4),
            "Peak_to_Peak": round(pp_a1, 6),
            "Max_Diff": max_diff,
            "Time_Period": f"{SLICE_DOWN} - {SLICE_UP}",
            "Remark": REMARK,
        },
        {
            "Rotation_Speed": ROTATE_SPEED_RPM,
            "Channel": "A2",
            "DC_Component": round(dc_a2, 6),
            "Main_Frequency_Hz": round(main_freq_a2, 2),
            "Main_Amplitude": round(main_amp_a2, 4),
            "Peak_to_Peak": round(pp_a2, 6),
            "Max_Diff": max_diff,
            "Time_Period": f"{SLICE_DOWN} - {SLICE_UP}",
            "Remark": REMARK,
        }
    ]
    result_df = pd.DataFrame(result_data)
    if WRITE_TO_CSV:
        # 【核心追加逻辑】文件存在则追加新数据，原有数据保留；不存在则新建表头
        save_path = r"C:\Users\wngan\Desktop\采集卡\fft_vibration_result.csv"
        if os.path.exists(save_path):
            # mode="a" 追加写入，header=False 不重复写表头
            result_df.to_csv(save_path, mode="a", header=False, index=False, encoding="utf-8-sig")
            print(f"\nFile existed, append {ROTATE_SPEED_RPM} rpm data to CSV.")
        else:
            # 新建文件，写入表头+当前转速数据
            result_df.to_csv(save_path, mode="w", header=True, index=False, encoding="utf-8-sig")
            print(f"\nNew CSV created, save {ROTATE_SPEED_RPM} rpm data.")

    # Plot figure: 2 rows 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Subplot 0,0: A1 Time Domain
    axes[0, 0].plot(x, y1, c="#e74c3c", linewidth=1.2)
    axes[0, 0].set_title("Channel A1 Time Domain (80~81s)")
    axes[0, 0].set_xlabel("Time t (s)")
    axes[0, 0].set_ylabel("Relative Displacement")
    axes[0, 0].grid(alpha=0.3)
    # X-axis margin left & right 5%
    t_min, t_max = x.min(), x.max()
    t_margin = (t_max - t_min) * 0.05
    axes[0, 0].set_xlim(t_min - t_margin, t_max + t_margin)

    # Subplot 0,1: A2 Time Domain
    axes[0, 1].plot(x, y2, c="#2980b9", linewidth=1.2)
    axes[0, 1].set_title("Channel A2 Time Domain (80~81s)")
    axes[0, 1].set_xlabel("Time t (s)")
    axes[0, 1].set_ylabel("Relative Displacement")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xlim(t_min - t_margin, t_max + t_margin)

    # Subplot 1,0: A1 FFT Spectrum
    axes[1, 0].plot(freq1, amp1, c="#e74c3c")
    axes[1, 0].set_title("Channel A1 Single-sided Amplitude Spectrum")
    axes[1, 0].set_xlabel("Frequency f (Hz)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(alpha=0.3)
    freq_max = fs1 / 2
    freq_margin = freq_max * 0.05
    axes[1, 0].set_xlim(-freq_margin, freq_max + freq_margin)

    # Subplot 1,1: A2 FFT Spectrum
    axes[1, 1].plot(freq2, amp2, c="#2980b9")
    axes[1, 1].set_title("Channel A2 Single-sided Amplitude Spectrum")
    axes[1, 1].set_xlabel("Frequency f (Hz)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlim(-freq_margin, freq_max + freq_margin)

    plt.tight_layout()
    plt.show()

def calc_peak_to_peak(signal):
    """
    计算信号峰峰值
    :param signal: 一维numpy数组时域信号
    :return: 峰峰值数值
    """
    if len(signal) < 2:
        return 0.0
    sig_max = np.max(signal)
    sig_min = np.min(signal)
    return sig_max - sig_min

def test():
    x = np.arange(0, 1.3, 0.01)
    y = 2 + np.sin(2 * np.pi * x)
    dc = np.mean(y)
    plt.plot(x, y)
    print("dc:", dc)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    read_data_and_plot()
    # test()