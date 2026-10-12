import math
import pandas as pd
import numpy as np
from fuzzy_logic import decide_use_nn
from composite_snr import compute_composite_snr_db


def get_velocity_vector(speed, direction):
    """
    将方向编码转换为二维速度向量。
    方向编码约定：
    0=西，1=北，2=东，3=南
    """
    direction_vectors = {
        0: (-1.0, 0.0),
        1: (0.0, 1.0),
        2: (1.0, 0.0),
        3: (0.0, -1.0)
    }

    unit_vector = direction_vectors.get(int(direction))
    if unit_vector is None:
        raise ValueError(f"未知方向编码: {direction}")

    return speed * unit_vector[0], speed * unit_vector[1]


def compute_relative_speed(tx_speed, tx_dir, rx_speed, rx_dir):
    """按二维速度向量计算相对速度大小。"""
    tx_vx, tx_vy = get_velocity_vector(tx_speed, tx_dir)
    rx_vx, rx_vy = get_velocity_vector(rx_speed, rx_dir)

    return math.sqrt((tx_vx - rx_vx) ** 2 + (tx_vy - rx_vy) ** 2)


def build_sample_seed(base_seed, time_value, tx_id):
    """为每个时间点和发送车构造稳定但不同的随机种子。"""
    tx_offset = sum(ord(ch) for ch in tx_id)
    return int(base_seed + int(time_value) * 1000 + tx_offset)


def process_random_cars_data(
    start_time=10,
    end_time=None,
    input_file="sim_data_two_way_crossroads.csv",
    output_file="random_cars_data.csv",
    random_seed=42
):
    """
    在指定时间范围内，针对 v1-v4 作为发送方，随机选择 5 辆车作为接收方，
    并输出与 nearest_cars_data.csv 相同结构的 CSV。
    """
    try:
        df_input = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return

    max_time_in_data = df_input["time"].max()

    if end_time is None:
        end_time = max_time_in_data

    if not (0 <= start_time < end_time <= max_time_in_data):
        print(f"错误：时间范围 [{start_time}, {end_time}] 不合法。")
        print(f"必须满足 0 <= start_time < end_time <= {max_time_in_data}。")
        return

    all_times = sorted(df_input["time"].unique())
    target_times = [t for t in all_times if start_time <= t <= end_time]
    target_tx_cars = ["v1", "v2", "v3", "v4"]

    results = []

    for t in target_times:
        time_data = df_input[df_input["time"] == t]

        for tx_id in target_tx_cars:
            tx_car = time_data[time_data["car_ID"] == tx_id]
            if tx_car.empty:
                continue

            tx_row = tx_car.iloc[0]
            tx_x, tx_y = tx_row["x_axis"], tx_row["y_axis"]
            tx_speed = tx_row["speed"]
            tx_dir = tx_row["direction"]
            tx_snr = tx_row["transmitter_SNR"]

            other_cars = time_data[time_data["car_ID"] != tx_id].copy()
            if other_cars.empty:
                continue

            other_cars["dist"] = np.sqrt((other_cars["x_axis"] - tx_x) ** 2 + (other_cars["y_axis"] - tx_y) ** 2)
            other_cars["rel_speed"] = other_cars.apply(
                lambda row: compute_relative_speed(
                    tx_speed=tx_speed,
                    tx_dir=tx_dir,
                    rx_speed=row["speed"],
                    rx_dir=row["direction"]
                ),
                axis=1
            )

            sample_size = min(5, len(other_cars))
            sample_seed = build_sample_seed(random_seed, t, tx_id)
            random_5 = other_cars.sample(n=sample_size, random_state=sample_seed)

            for _, rx_row in random_5.iterrows():
                res_row = {
                    "time": t,
                    "car_ID_TX": tx_id,
                    "car_ID_RX": rx_row["car_ID"],
                    "snr_values": float(tx_snr),
                    "distance_values": float(rx_row["dist"]),
                    "rel_speed_values": float(rx_row["rel_speed"])
                }

                res_row["snr_values_norm"] = (res_row["snr_values"] - (-10)) / (20 - (-10))
                res_row["distance_values_norm"] = (res_row["distance_values"] - 0) / (200 - 0)
                res_row["rel_speed_values_norm"] = (res_row["rel_speed_values"] - 0) / (100 - 0)

                res_row["composite_snr_db"] = compute_composite_snr_db(
                    res_row["snr_values"],
                    res_row["distance_values"],
                    res_row["rel_speed_values"]
                )

                res_row["use_nn"] = decide_use_nn(
                    res_row["snr_values_norm"],
                    res_row["distance_values_norm"],
                    res_row["rel_speed_values_norm"]
                )

                results.append(res_row)

    df_results = pd.DataFrame(results)

    if df_results.empty:
        print("未生成任何结果数据。")
        return

    df_results["snr_values"] = df_results["snr_values"].round(2)
    df_results["distance_values"] = df_results["distance_values"].round(2)
    df_results["rel_speed_values"] = df_results["rel_speed_values"].round(2)

    df_results["snr_values_norm"] = df_results["snr_values_norm"].round(4)
    df_results["distance_values_norm"] = df_results["distance_values_norm"].round(4)
    df_results["rel_speed_values_norm"] = df_results["rel_speed_values_norm"].round(4)
    df_results["composite_snr_db"] = df_results["composite_snr_db"].round(4)

    cols = [
        "time", "car_ID_TX", "car_ID_RX", "snr_values", "distance_values", "rel_speed_values",
        "snr_values_norm", "distance_values_norm", "rel_speed_values_norm", "composite_snr_db", "use_nn"
    ]
    df_results.to_csv(output_file, index=False, columns=cols)
    print(f"数据处理完成，结果已保存至 {output_file}")


if __name__ == "__main__":
    process_random_cars_data(start_time=10, end_time=70)
