import streamlit as st
import numpy as np
import polars as pl
import pandas as pd
from io import StringIO
import json

# Đọc tệp user_list.txt và lọc người dùng theo điều kiện
def load_filtered_users(user_list_file_path):
    filtered_users = {}
    try:

        with open(user_list_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Bỏ qua dòng tiêu đề
                if 'org_id' in line or 'remap_id' in line:
                    continue
                parts = line.strip().split()
                if len(parts) >= 13:  # Đảm bảo đủ cột dữ liệu
                    org_id = parts[0]  # Không chuyển đổi org_id
                    try:
                        remap_id = int(parts[1])  # remap_id là số nguyên
                        sinhvien_nam = int(parts[12])  # Lọc theo sinhvien_nam
                        filtered_users[org_id] = (remap_id, sinhvien_nam)
                    except ValueError:
                        print(f"Không thể chuyển đổi giá trị trong dòng: {line}")
    except FileNotFoundError:
        print(f"Tệp {user_list_file_path} không tồn tại.")
    return filtered_users

# Hàm lấy top 5 items tương tự như trước nhưng có ánh xạ từ filtered_users
def get_top_5_items_filtered(file_path, filtered_users, item_mapping_file_path, mssv_input):
    try:
        scores = np.load(file_path)
    except FileNotFoundError:
        print(f"Tệp {file_path} không tồn tại.")
        return []

    user_index = filtered_users[mssv_input][0]

    if len(scores) <= user_index:
        print("Chỉ số user không hợp lệ.")
        return []

    user_scores = scores[user_index]  # Lấy điểm của user
    top_5_items = np.argsort(user_scores)[::-1][:5]  # Lấy 5 item có điểm số cao nhất

    # Tải ánh xạ item
    remap_dict = {}
    try:
        with open(item_mapping_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'remap_id' in line or 'org_id' in line:
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        remap_id = int(parts[1])
                        remap_dict[remap_id] = parts[0]
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Tệp ánh xạ {item_mapping_file_path} không tồn tại.")

    result = []
    for item_index in top_5_items:
        item_name = remap_dict.get(item_index, f"Item {item_index + 1}")
        result.append((item_name, user_scores[item_index]))

    return result

# Hiển thị top 5 môn học dựa trên file_path cụ thể.
def display_top_5(file_path, filtered_users, item_mapping_file_path, mssv_input, title):
    top_5_items = get_top_5_items_filtered(file_path, filtered_users, item_mapping_file_path, mssv_input)

    if top_5_items:
        st.write(f"{title}:")
        data = pl.DataFrame({
            "Mã môn học": [item[0].upper() for item in top_5_items],
            "Scores": [item[1] for item in top_5_items],
        })

        st.table(data)
    else:
        st.write(f"{title}: Không có dữ liệu điểm hoặc sinh viên không đủ điều kiện.")

# Hàm để mã hóa mssv_input với dữ liệu trong mssv.json và lấy mssv_raw
def decode_mssv(mssv_input, mssv_json_file='./mssv.json'):
    try:
        # Đọc dữ liệu từ file mssv.json
        with open(mssv_json_file, 'r', encoding='utf-8') as f:
            mssv_data = json.load(f)

        # Kiểm tra nếu mssv_input có trong file
        for value in mssv_data.values():
            for item in value:
                st.sidebar.warning(f"value {value} item {item}")
                if item['mssv'] == mssv_input:
                    mssv_raw = item['mssv_raw']
                    print(mssv_raw)
                    return mssv_raw
        else:
            st.sidebar.warning(f"MSSV {mssv_input} không có trong dữ liệu.")
            return mssv_input
    except FileNotFoundError:
        print(f"File {mssv_json_file} không tồn tại.")
        return None
    except json.JSONDecodeError:
        print(f"File {mssv_json_file} không phải là file JSON hợp lệ.")
        return None


# Hàm run để xử lý toàn bộ quá trình
def run(mssv_input):
    # Đường dẫn đến các tệp dữ liệu
    user_list_file_path = "../notebooks/Model/KGAT-pytorch/KGAT_data/user_list.txt"
    item_mapping_file_path = "../Data\Train_test_data\Data_mapping\mp_mamh.txt"

    # Tải danh sách người dùng thỏa mãn điều kiện
    filtered_users = load_filtered_users(user_list_file_path)

    mssv_begin = mssv_input
    if mssv_input != "":
        mssv_input = decode_mssv(mssv_input)

    if mssv_input == "":
        # Nếu MSSV chưa nhập, hiển thị yêu cầu nhập MSSV
        st.sidebar.success("Vui lòng nhập mã số sinh viên để tiếp tục.")
    elif mssv_input not in filtered_users:
        # Nếu MSSV không hợp lệ
        st.sidebar.warning(f"MSSV {mssv_input} không đủ điều kiện.")
    else:
        # Kiểm tra sinhvien_nam >= 3
        sinhvien_nam = filtered_users.get(mssv_input)[1]
        if sinhvien_nam is not None and sinhvien_nam < 3:
            st.sidebar.warning(f"MSSV {mssv_input} không đủ điều kiện (sinhvien_nam < 3).")
            return

        # Hiển thị giao diện Streamlit
        st.title(f'Môn học đề xuất cho sinh viên {mssv_begin}')
        file_paths = [
            (
                r"../notebooks\Model\KGAT-pytorch\trained_model\BPRMF\KGAT_data\embed-dim64_lr0.0001_pretrain2\cf_scores_bprmf.npy",
                "Matrix Factorization (BPRMF)"),
            (
                r"../notebooks\Model\KGAT-pytorch\trained_model\NFM\KGAT_data\fm_embed-dim64_64-32-16_lr0.0001_pretrain2_user-info0\cf_scores_fm0.npy",
                "Factorization Machine without user_info"),
            (
                r"../notebooks\Model\KGAT-pytorch\trained_model\NFM\KGAT_data\fm_embed-dim64_64-32-16_lr0.0001_pretrain2_user-info1\cf_scores_fm1.npy",
                "Factorization Machine with user_info"),
            (
                r"../notebooks\Model\KGAT-pytorch\trained_model\NFM\KGAT_data\nfm_embed-dim64_64-32-16_lr0.0001_pretrain2_user-info0\cf_scores_nfm0.npy",
                "Neural Factorization Machine (NFM) without user_ìnfo"),
            (
                r"../notebooks\Model\KGAT-pytorch\trained_model\NFM\KGAT_data\nfm_embed-dim64_64-32-16_lr0.0001_pretrain2_user-info0\cf_scores_nfm0.npy",
                "Neural Factorization Machine (NFM) with user_info"),
            (
                r"../notebooks\Model\KGAT-pytorch\trained_model\KGAT\KGAT_data\embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain2\cf_scores_kgat.npy",
                "KGAT"),
            (
                r"../notebooks\Model\KGAT-pytorch\trained_model\KGAT\KGAT_data\embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain2\cf_scores_embed.npy",
                "KGAT EMBED"),
        ]

        for file_path, title in file_paths:
            display_top_5(file_path, filtered_users, item_mapping_file_path, mssv_input, title)


if __name__ == "__main__":
    # Sidebar để nhập MSSV
    st.sidebar.header("Đăng nhập")
    mssv_input = st.sidebar.text_input("Nhập mã số sinh viên:", value="")

    run(mssv_input)
