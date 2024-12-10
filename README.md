# ReadMe.txt
## Tác giả:
"Hồ Vũ An - 21521804"  
"Dương Uy Quan - 21521323"  
"Lê Thanh Phong - 21521271"  
"Nguyễn Thành Trung - 21521595"  
"Phạm Tiến Đạt - 21521948"  
***
## 1. Giới thiệu chung
Đây là đồ án cho môn học IS353 - Mạng xã hội - Nhóm 6
Đồ án này sẽ thực hiện các chạy các mô hình như là KGAT, Factorization Machine, Neural Factorization Machine, Bayesian Personalized Ranking Matrix Factorization trên bộ dữ liệu uit(được cung cấp).
Đồ án có sự tham khảo bởi 2 đồ án khác:
        [Hệ thống đề xuất cho MOOC](https://drive.google.com/drive/folders/1DZPvcWluXIlHK-LyXls4Ej4ze8BPH2yo), 
        [Hệ thống đề xuất cho UIT](https://drive.google.com/drive/folders/1E2zw9jS5222i8XKKGzWYlRsALtJYvLX_)  
Các mô hình sử dụng được lấy từ link: https://github.com/LunaBlack/KGAT-pytorch?tab=readme-ov-file#environment-requirement
***
## 2. Các mô hình trong dự án
### 2.1. KGAT (Knowledge Graph Attention Network)
KGAT là một mô hình học sâu kết hợp giữa mạng nơ-ron và đồ thị tri thức. Nó được thiết kế để xử lý các bài toán học máy với dữ liệu có cấu trúc đồ thị. Các tính năng nổi bật của KGAT:

Kết hợp các thông tin về người dùng và sản phẩm từ đồ thị tri thức.
Sử dụng cơ chế Attention để học được các trọng số quan trọng của các nút trong đồ thị.
Phù hợp với các bài toán gợi ý trong đó các đối tượng có quan hệ với nhau thông qua đồ thị (ví dụ: người dùng, sản phẩm, danh mục).
### 2.2. FM (Factorization Machine)
FM là một mô hình học máy được thiết kế để giải quyết các bài toán học máy trong môi trường có nhiều đặc tính tương tác (ví dụ: các đặc tính của người dùng và sản phẩm). FM có khả năng học được mối quan hệ không tuyến tính giữa các đặc tính, điều này giúp mô hình hoạt động tốt trong các bài toán gợi ý.

FM có thể học các mối quan hệ giữa các đặc tính mà không cần tính toán quá phức tạp, rất hiệu quả cho bài toán hệ thống đề xuất.
Mô hình này giúp giảm thiểu số lượng các tham số so với các mô hình khác như mạng nơ-ron, mà vẫn đạt được hiệu suất khá cao.
### 2.3. NFM (Neural Factorization Machine)
NFM là một mở rộng của mô hình FM nhưng kết hợp với mạng nơ-ron để tăng cường khả năng học các mối quan hệ không tuyến tính. Trong NFM, các tính năng của người dùng và sản phẩm được kết hợp qua một mạng nơ-ron để học các tương tác phức tạp hơn.

NFM kết hợp các đặc điểm của FM và học sâu (deep learning).
Mô hình này có khả năng cải thiện độ chính xác của dự đoán gợi ý nhờ vào việc sử dụng mạng nơ-ron để học các biểu diễn ẩn của các đặc tính.
### 2.4. BPRMF (Bayesian Personalized Ranking Matrix Factorization)
BPRMF là một mô hình ma trận phân rã được sử dụng để tối ưu hóa các hệ thống đề xuất dựa trên Bayesian Personalized Ranking. BPRMF tối ưu hóa các thứ hạng của các mục trong hệ thống gợi ý thay vì chỉ đơn giản là dự đoán điểm số của người dùng đối với các mục.

BPRMF sử dụng phương pháp Bayesian để tối ưu hóa thứ hạng của các đối tượng trong ma trận phân rã.
Mô hình này rất hiệu quả trong các bài toán gợi ý mà mục tiêu là xếp hạng các sản phẩm thay vì dự đoán điểm số chính xác.
***
# 3. Môi trường
code được chạy với Python 3.9 và các thư viện liên quan:
```
torch == 2.5.1
numpy == 2.1.1
pandas == 1.3.5
scipy == 1.14.1
polars == 1.17.0
tqdm == 4.67.0
scikit-learn == 1.5.2
```
Chi tiết trong file requirements.txt
***
4. Chạy code
- Train
```
python file_main_mô_hình.py --model_type fm --data_name amazon-book
```
- Predict
```
python file_predict_mô_hình.py --model_type fm --data_name amazon-book
```
