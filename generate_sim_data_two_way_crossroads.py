import csv
import random


# 方向编码：0=西，1=北，2=东，3=南
DIRECTION_WEST = 0
DIRECTION_NORTH = 1
DIRECTION_EAST = 2
DIRECTION_SOUTH = 3


def get_position_by_route(route, distance, road_length):
    """
    根据路线和累计行驶距离，计算车辆在十字路口道路中的位置和方向。

    坐标原点设在十字路口中心 (0, 0)，每条道路总长度为 road_length，
    因此每个方向的入口位于坐标轴两端的 +/- road_length / 2 位置。
    """
    half_length = road_length / 2
    lane_center = 1.5

    if route == "bottom_to_top":
        x_axis = -lane_center
        y_axis = -half_length + distance
        direction = DIRECTION_NORTH
    elif route == "top_to_bottom":
        x_axis = lane_center
        y_axis = half_length - distance
        direction = DIRECTION_SOUTH
    elif route == "left_to_right":
        x_axis = -half_length + distance
        y_axis = -lane_center
        direction = DIRECTION_EAST
    elif route == "right_to_left":
        x_axis = half_length - distance
        y_axis = lane_center
        direction = DIRECTION_WEST
    else:
        raise ValueError(f"未知路线类型: {route}")

    return round(x_axis, 2), round(y_axis, 2), direction


def generate_car_sim_data_crossroads(number_cars=20, road_length=2000, road_width=6):
    """
    生成十字路口车辆行驶模拟数据并保存为 CSV 文件。

    参数:
    number_cars (int): 车辆总数 (>= 4)
    road_length (int): 每条道路总长度 (>= 100 且为 100 的倍数)
    road_width (int): 道路宽度，当前脚本固定按 6m 双向 2 车道建模
    """
    if number_cars < 4:
        print("错误：number_cars 必须大于等于 4")
        return

    if road_width != 6:
        print("错误：当前十字路口脚本固定按 6m 宽双向 2 车道建模，road_width 必须为 6")
        return

    if road_length < 100 or road_length % 100 != 0:
        print("错误：road_length 必须大于等于 100 且为 100 的倍数")
        return

    cars = []
    speeds_pool = [10, 20, 30, 40, 50]
    snr_pool = [20, 30, 40]
    random.seed(42)

    routes = [
        "bottom_to_top",
        "top_to_bottom",
        "left_to_right",
        "right_to_left"
    ]

    for i in range(number_cars):
        car_id = f"v{i + 1}"
        route = routes[i % len(routes)]
        x_axis, y_axis, direction = get_position_by_route(
            route=route,
            distance=0,
            road_length=road_length
        )
        snr = random.choice(snr_pool)

        cars.append({
            "car_ID": car_id,
            "route": route,
            "distance": 0.0,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "direction": direction,
            "transmitter_SNR": snr,
            "finished": False
        })

    filename = "sim_data_two_way_crossroads.csv"
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
                        car["distance"] = min(car["distance"] + speed, road_length)
                        car["x_axis"], car["y_axis"], car["direction"] = get_position_by_route(
                            route=car["route"],
                            distance=car["distance"],
                            road_length=road_length
                        )

                        if car["distance"] >= road_length:
                            car["finished"] = True

                t += 1

        print(f"成功生成十字路口模拟数据并保存到 {filename}")
        print(f"总时间点数: {t}")

    except Exception as e:
        print(f"写入 CSV 文件时出错: {e}")


if __name__ == "__main__":
    # 测试用例：20辆车，十字路口每条道路长度 2000m，宽度 6m
    generate_car_sim_data_crossroads(number_cars=20, road_length=2000, road_width=6)
