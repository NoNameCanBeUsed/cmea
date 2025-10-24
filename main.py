import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def generate_ddt(sbox):
    n = len(sbox)
    max_ddt = 0
    ddt = np.zeros((n, n), dtype=int)
    for x in range(n):
        for dx in range(n):
            dy = sbox[x] ^ sbox[x ^ dx]
            ddt[dx][dy] += 1
    ddt_r = ddt.ravel()
    for index in range(len(ddt_r)):
        if max_ddt < ddt_r[index] and ddt_r[index] < 256 :
            max_ddt = ddt_r[index]
    return ddt#,max_ddt

def generate_lat(sbox):
    n = len(sbox)
    max_lat = 0
    lat = np.zeros((n, n), dtype=int)
    for a in range(n):
        for b in range(n):
            count = 0
            for x in range(n):
                if bin((a & x)).count('1') % 2 == bin((b & sbox[x])).count('1') % 2:
                    count += 1
            lat[a][b] = (count * 2 - n)/2
    lat_r = lat.ravel()
    for index in range(len(lat_r)):
        if max_lat < abs(lat_r[index]) and abs(lat_r[index]) < 128  :
            max_lat = abs(lat_r[index])
    return lat#,max_lat

def generate_abslat(sbox):
    n = len(sbox)
    max_lat = 0
    lat = np.zeros((n, n), dtype=int)
    for a in range(n):
        for b in range(n):
            count = 0
            for x in range(n):
                if bin((a & x)).count('1') % 2 == bin((b & sbox[x])).count('1') % 2:
                    count += 1
            lat[a][b] = abs((count * 2 - n)/2)
    lat_r = lat.ravel()
    for index in range(len(lat_r)):
        if max_lat < abs(lat_r[index]) and abs(lat_r[index]) < 128  :
            max_lat = abs(lat_r[index])
    return lat

def print_table(table, name):
    print(f"{name}:")
    for row in table:
        print(" ".join(f"{val:3}" for val in row))
    print()
def print_table2(table, name):
    print(f"{name}:")
    for i in range(0, len(table), 16):
        row = table[i]
        for j in range(0, len(row), 16):
            print(f"{row[j]:3}", end=" ")
        print()
    print()
if __name__ == "__main__":
    # 原始Cmea S盒（8比特输入→8比特输出，一维列表形式）
    cmea_sbox = [
        0xd9, 0x23, 0x5f, 0xe6, 0xca, 0x68, 0x97, 0xb0, 0x7b, 0xf2, 0x0c, 0x34, 0x11, 0xa5, 0x8d, 0x4e,
        0x0a, 0x46, 0x77, 0x8d, 0x10, 0x9f, 0x5e, 0x62, 0xf1, 0x34, 0xec, 0xa5, 0xc9, 0xb3, 0xd8, 0x2b,
        0x59, 0x47, 0xe3, 0xd2, 0xff, 0xae, 0x64, 0xca, 0x15, 0x8b, 0x7d, 0x38, 0x21, 0xbc, 0x96, 0x00,
        0x49, 0x56, 0x23, 0x15, 0x97, 0xe4, 0xcb, 0x6f, 0xf2, 0x70, 0x3c, 0x88, 0xba, 0xd1, 0x0d, 0xae,
        0xe2, 0x38, 0xba, 0x44, 0x9f, 0x83, 0x5d, 0x1c, 0xde, 0xab, 0xc7, 0x65, 0xf1, 0x76, 0x09, 0x20,
        0x86, 0xbd, 0x0a, 0xf1, 0x3c, 0xa7, 0x29, 0x93, 0xcb, 0x45, 0x5f, 0xe8, 0x10, 0x74, 0x62, 0xde,
        0xb8, 0x77, 0x80, 0xd1, 0x12, 0x26, 0xac, 0x6d, 0xe9, 0xcf, 0xf3, 0x54, 0x3a, 0x0b, 0x95, 0x4e,
        0xb1, 0x30, 0xa4, 0x96, 0xf8, 0x57, 0x49, 0x8e, 0x05, 0x1f, 0x62, 0x7c, 0xc3, 0x2b, 0xda, 0xed,
        0xbb, 0x86, 0x0d, 0x7a, 0x97, 0x13, 0x6c, 0x4e, 0x51, 0x30, 0xe5, 0xf2, 0x2f, 0xd8, 0xc4, 0xa9,
        0x91, 0x76, 0xf0, 0x17, 0x43, 0x38, 0x29, 0x84, 0xa2, 0xdb, 0xef, 0x65, 0x5e, 0xca, 0x0d, 0xbc,
        0xe7, 0xfa, 0xd8, 0x81, 0x6f, 0x00, 0x14, 0x42, 0x25, 0x7c, 0x5d, 0xc9, 0x9e, 0xb6, 0x33, 0xab,
        0x5a, 0x6f, 0x9b, 0xd9, 0xfe, 0x71, 0x44, 0xc5, 0x37, 0xa2, 0x88, 0x2d, 0x00, 0xb6, 0x13, 0xec,
        0x4e, 0x96, 0xa8, 0x5a, 0xb5, 0xd7, 0xc3, 0x8d, 0x3f, 0xf2, 0xec, 0x04, 0x60, 0x71, 0x1b, 0x29,
        0x04, 0x79, 0xe3, 0xc7, 0x1b, 0x66, 0x81, 0x4a, 0x25, 0x9d, 0xdc, 0x5f, 0x3e, 0xb0, 0xf8, 0xa2,
        0x91, 0x34, 0xf6, 0x5c, 0x67, 0x89, 0x73, 0x05, 0x22, 0xaa, 0xcb, 0xee, 0xbf, 0x18, 0xd0, 0x4d,
        0xf5, 0x36, 0xae, 0x01, 0x2f, 0x94, 0xc3, 0x49, 0x8b, 0xbd, 0x58, 0x12, 0xe0, 0x77, 0x6c, 0xda
    ]
    #extra_sbox= [0xe,0x4,0xd,0x1,0x2,0xf,0xb,0x8,0x3,0xa,0x6,0xc,0x5,0x9,0x0,0x7]

    def branch_swap_W(x):
        """交换8比特输入的高4比特和低4比特"""
        high = (x >> 4) & 0x0F  # 提取高4比特
        low = x & 0x0F  # 提取低4比特
        return (low << 4) | high  # 交换后重组

    sbox = [
        cmea_sbox[branch_swap_W(x)] for x in range(256)
    ]
    sbox2 = [
        branch_swap_W(cmea_sbox[y]) for y in range(256)
    ]
    #sbox =cmea_sbox
   # ddt,maxddt = generate_ddt(sbox)
    #lat,maxlat = generate_lat(cmea_sbox)
    #lat2, maxlat2 = generate_lat(sbox2)
    #np.set_printoptions(threshold=np.inf)
    #print(maxddt)
    #print(maxlat)
    #print_table(ddt, "DDT")
    #print_table(lat, "LAT")
    #print_table2(lat, "LAT2")
    #print_table(lat2, "LAT2")
'''
以下是加入的新代码

'''


def tu_core_decomposition(f):
    n = 4
    size = 2 ** n  #
    T = [[0 for _ in range(size)] for __ in range(size)]
    U = [[0 for _ in range(size)] for __ in range(size)]

    for y in range(size):
        left_outputs = []
        for x in range(size):
            input_8bit = (x << n) | y
            output_8bit = f(input_8bit)
            left_4bit = (output_8bit >> n) & 0x0F
            left_outputs.append(left_4bit)
        for x in range(size):
            T[y][x] = left_outputs[x]
    for y in range(size):
        T_inv = [0] * size
        for x in range(size):
            k = T[y][x]
            T_inv[k] = x
        for k in range(size):
            x = T_inv[k]
            input_8bit = (x << n) | y
            output_8bit = f(input_8bit)
            right_4bit = output_8bit & 0x0F
            U[k][y] = right_4bit
    return T, U
def branch_swap(x):
    high4 = (x >> 4) & 0x0F
    low4 = x & 0x0F
    return (low4 << 4) | high4
if __name__ == "__main__":
    def sbox_with_W(x):
        swapped_x = branch_swap(x)
        return sbox2[swapped_x]
    print("TUcore-Decomposition启动...")
    T, U = tu_core_decomposition(sbox_with_W)
    print("\n===== 分解结果示例 =====")
    print("\nT：")
    for row in T:
        print(" ".join([f"0x{v:01x}" for v in row]))
    t_box = [
        abs(tbox) for row in T for tbox in row
    ]
    print("\nU：")
    for row in U:
        print(" ".join([f"0x{v:01x}" for v in row]))
    u_box = [
        branch_swap(ubox) for row in U for ubox in row
    ]
    t_lat = generate_abslat(t_box)
    u_lat = generate_abslat(u_box)
    #print("T lat:")
    #print_table(t_lat,"Tlat")
    #plt.imshow(t_lat,cmap="Reds")
    #plt.colorbar()  # 添加颜色条
    #plt.title("TBox LAT")
    #print("U lat:")
    print_table(u_lat,"Ulat")
    plt.imshow(u_lat, cmap="Greys")
    plt.colorbar()  # 添加颜色条
    plt.title("uBox LAT")
    plt.show()
