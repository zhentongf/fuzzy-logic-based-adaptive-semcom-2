import csv
import random


# 方向编码：0=西，1=北，2=东，3=南
DIRECTION_WEST = 0
DIRECTION_NORTH = 1
DIRECTION_EAST = 2
DIRECTION_SOUTH = 3


def get_lane_offsets(road_width):
    """按 3m 一个车道生成车道中心偏移量。"""
    lane_count = road_width // 3
    return [1.5 + 3 * i for i in range(lane_count)]


def get_position_and_direction(distance, road_length, lane_offset):
    """
    根据累计行驶距离，计算车辆在方形环路上的位置和方向。

    以东南角为原点 (0, 0)，车辆按顺时针行驶：
    向西 -> 向北 -> 向东 -> 向南。
    为了让 9m 宽道路中的不同车道形成连续闭环，每条车道使用一个内缩的方形路径。
    """
    side_length = road_length - 2 * lane_offset
    perimeter = 4 * side_length

    if distance >= perimeter:
        return round(-lane_offset, 2), round(lane_offset, 2), DIRECTION_WEST

    if distance < side_length:
        x_axis = -lane_offset - distance
        y_axis = lane_offset
        direction = DIRECTION_WEST
    elif distance < 2 * side_length:
        segment_distance = distance - side_length
        x_axis = -(road_length - lane_offset)
        y_axis = lane_offset + segment_distance
        direction = DIRECTION_NORTH
    elif distance < 3 * side_length:
        segment_distance = distance - 2 * side_length
        x_axis = -(road_length - lane_offset) + segment_distance
        y_axis = road_length - lane_offset
        direction = DIRECTION_EAST
    else:
        segment_distance = distance - 3 * side_length
        x_axis = -lane_offset
        y_axis = road_length - lane_offset - segment_distance
        direction = DIRECTION_SOUTH

    return round(x_axis, 2), round(y_axis, 2), direction


def generate_car_sim_data_round(number_cars=20, road_length=1000, road_width=9):
    """
    生成环形车辆行驶模拟数据并保存为 CSV 文件。

    参数:
    number_cars (int): 车辆总数 (>= 1)
    road_length (int): 每条直线段长度 (>= 100 且为 100 的倍数)
    road_width (int): 道路宽度 (>= 3 且为 3 的倍数)
    """
    if number_cars < 1:
        print("错误：number_cars 必须大于等于 1")
        return

    if road_width < 3 or road_width % 3 != 0:
        print("错误：road_width 必须大于等于 3，且为 3 的整数倍")
        return

    if road_length < 100 or road_length % 100 != 0:
        print("错误：road_length 必须大于等于 100 且为 100 的倍数")
        return

    lane_offsets = get_lane_offsets(road_width)
    if not lane_offsets:
        print("错误：无法根据 road_width 生成有效车道")
        return

    max_lane_offset = max(lane_offsets)
    if road_length <= 2 * max_lane_offset:
        print("错误：road_length 过短，无法生成闭环车道")
        return

    cars = []
    speeds_pool = [10, 20, 30, 40, 50]
    snr_pool = [20, 30, 40]
    random.seed(42)

    for i in range(number_cars):
        car_id = f"v{i + 1}"
        lane_offset = lane_offsets[i % len(lane_offsets)]
        x_axis, y_axis, direction = get_position_and_direction(
            distance=0,
            road_length=road_length,
            lane_offset=lane_offset
        )
        lap_length = 4 * (road_length - 2 * lane_offset)
        snr = random.choice(snr_pool)

        cars.append({
            "car_ID": car_id,
            "lane_offset": lane_offset,
            "distance": 0.0,
            "lap_length": lap_length,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "direction": direction,
            "transmitter_SNR": snr,
            "finished": False
        })

    filename = "sim_data_two_way_round_square.csv"
    headers = ["time", "car_ID", "x_axis", "y_axis", "speed", "direction", "transmitter_SNR"]

    try:
        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            t = 0
            while True:
                active_cars = [car for car in cars if not car["finished"]]
                if not active_cars:
                    break

                for i, car in enumerate(cars):
                    is_started = t >= i

                    if car["finished"] or not is_started:
                        speed = 0
                    else:
                        speed = random.choice(speeds_pool)

                    writer.writerow({
                        "time": t,
                        "car_ID": car["car_ID"],
                        "x_axis": car["x_axis"],
                        "y_axis": car["y_axis"],
                        "speed": speed,
                        "direction": car["direction"],
                        "transmitter_SNR": car["transmitter_SNR"]
                    })

                    if is_started and not car["finished"]:
                        car["distance"] = min(car["distance"] + speed, car["lap_length"])
                        car["x_axis"], car["y_axis"], car["direction"] = get_position_and_direction(
                            distance=car["distance"],
                            road_length=road_length,
                            lane_offset=car["lane_offset"]
                        )

                        if car["distance"] >= car["lap_length"]:
                            car["finished"] = True

                t += 1

        print(f"成功生成环形模拟数据并保存到 {filename}")
        print(f"总时间点数: {t}")

    except Exception as e:
        print(f"写入 CSV 文件时出错: {e}")


if __name__ == "__main__":
    # 测试用例：20辆车，四段各 1000m，车道总宽 9m
    generate_car_sim_data_round(number_cars=20, road_length=1000, road_width=9)
