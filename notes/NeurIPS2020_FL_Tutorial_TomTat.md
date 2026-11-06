# Tóm tắt: Federated Learning Tutorial – NeurIPS 2020

**Diễn giả:** Peter Kairouz, Brendan McMahan (Google), Virginia Smith (CMU)  
**Nguồn:** slide tutorial (https://sites.google.com/view/fl-tutorial)  
**Tài liệu gốc nên đọc kèm:** Kairouz et al., *Advances and Open Problems in Federated Learning* (arXiv:1912.04977)

Tutorial gồm 4 phần: (I) FL là gì, (II) Federated optimization, (III) Privacy, (IV) Bài toán mở. Cuối mỗi phần có ghi chú liên hệ với [lộ trình PhD 3 năm](../AiTA_Lab_PhD_FL_Research_Roadmap_3_Years.md).

---

## Phần I – Federated Learning là gì?

### Động lực
Dữ liệu sinh ra ở biên (điện thoại, IoT, bệnh viện). Xử lý trên thiết bị giúp giảm độ trễ, chạy offline, tiết kiệm pin và có lợi về privacy. Câu hỏi đặt ra: nếu dữ liệu ở lại biên thì có huấn luyện mô hình được không?

### Định nghĩa (theo Kairouz et al.)
> FL là thiết lập học máy trong đó nhiều thực thể (client) cùng giải một bài toán dưới sự điều phối của một server. Dữ liệu thô của mỗi client được giữ tại chỗ, không trao đổi; thay vào đó chỉ gửi các **cập nhật tập trung (focused updates)** để **tổng hợp ngay**.

### Ứng dụng thực tế
- **Cross-device:** Gboard (dự đoán từ tiếp theo, accuracy +24% so với n-gram; emoji; phát hiện từ mới), Apple Siri (nhận dạng người nói).
- **Cross-silo:** Intel + UPenn (29 tổ chức y tế, phát hiện u não), NVIDIA Clara (đánh giá ảnh nhũ ảnh/mammogram).

Ứng dụng tốt cho FL khi: dữ liệu trên thiết bị phù hợp hơn dữ liệu proxy ở server, dữ liệu nhạy cảm hoặc lớn, và nhãn có thể suy ra tự nhiên từ tương tác người dùng.

### So sánh ba thiết lập

| Tiêu chí | Distributed (datacenter) | **Cross-silo FL** | **Cross-device FL** |
|---|---|---|---|
| Client | Node trong một cluster | Tổ chức (bệnh viện, ngân hàng) | Điện thoại / IoT |
| Quy mô | 1–1000 | **2–100** | tới 10^10 |
| Phân phối dữ liệu | Có thể xáo trộn, IID | Non-IID, cố định | Non-IID, cố định |
| Sẵn sàng | Luôn sẵn sàng | Hầu như luôn sẵn sàng | Chỉ một phần nhỏ online |
| Địa chỉ hóa / trạng thái | Có ID, stateful | **Có ID, stateful** (tham gia mọi vòng) | Không ID, **stateless** (thường chỉ tham gia 1 lần) |
| Nút thắt | Tính toán | Tính toán hoặc truyền thông | Truyền thông |
| Độ tin cậy | Ít lỗi | Ít lỗi | ≥5% client rớt mỗi vòng |
| Phân hoạch | Tùy ý | Ngang (horizontal) hoặc dọc (vertical) | Ngang |

FL khác **decentralized (peer-to-peer) learning** ở chỗ FL có server điều phối (hub-and-spoke), còn P2P không có.

### Quy trình cross-device và nguyên tắc privacy
Thiết bị đủ điều kiện (sạc, wifi, rảnh) check-in → nhận mô hình → huấn luyện cục bộ → gửi cập nhật → server tổng hợp. Ba nguyên tắc: **focused collection** (chỉ gửi đúng thứ cần), **ephemeral reports** (server không lưu báo cáo từng thiết bị), **only-in-aggregate** (kỹ sư chỉ thấy kết quả tổng hợp).

Quy mô điển hình: 100–1000 client/vòng, hàng nghìn vòng, 1–10 phút/vòng.

### Federated Analytics
Áp dụng cùng ý tưởng cho thống kê/khoa học dữ liệu (histogram, quantile, heavy hitters, federated SQL…), chỉ công bố kết quả tổng hợp.

> **Liên hệ lộ trình:** Bảng so sánh trên chính là cơ sở cho mục 1.4 của lộ trình (MIA là cross-silo, stateful, ít client, tham gia đầy đủ → client selection không phải trọng tâm; weighting/aggregation và personalization mới là trọng tâm).

---

## Phần II – Federated Optimization

### Bài toán
Tối thiểu hóa rủi ro thực nghiệm có trọng số trên m thiết bị:

  min_w F(w) = Σ_k p_k F_k(w),  với F_k là loss trung bình trên dữ liệu cục bộ của client k (thường p_k = n_k / n).

Bốn thách thức: mạng lớn và chậm; ràng buộc privacy; phần cứng/kết nối đa dạng (**system heterogeneity**); dữ liệu non-IID, mất cân bằng (**statistical heterogeneity**).

### FedAvg (McMahan et al., AISTATS 2017)
Mỗi vòng: chọn tập con client → mỗi client chạy E epoch SGD cục bộ → server lấy trung bình có trọng số các cập nhật.
- Khác distributed SGD ở chỗ **local updating**: tính nhiều bước cục bộ trước khi giao tiếp, giảm số vòng giao tiếp tới ~100 lần.
- Local updating không mới (one-shot averaging, ADMM, CoCoA, Local SGD), nhưng FL khác biệt vì kết hợp **local updating + dữ liệu không đồng nhất + tham gia một phần + mục tiêu phi lồi**.
- Nhược điểm: khi dữ liệu càng non-IID và client bị rớt (system heterogeneity), FedAvg hội tụ chậm, dao động, thậm chí phân kỳ.

### FedProx (Li et al., MLSys 2020)
- Thêm **proximal term** (μ/2)‖w − w^t‖² vào loss cục bộ để giới hạn mức “trôi” của client.
- Không bỏ client chậm mà chấp nhận phần công việc đã làm (partial work).
- Có bảo đảm hội tụ (lồi và phi lồi) dưới giả định **B-dissimilarity** (độ khác biệt gradient bị chặn).

### Giảm chi phí truyền thông
Ba hướng: (1) ít client mỗi vòng, (2) ít vòng hơn, (3) thông điệp nhỏ hơn. Với (3): low-rank, sparsification, quantization, dropout; áp dụng cả uplink lẫn downlink. Thực nghiệm EMNIST: lượng tử hóa 4 bit gần như không mất accuracy. Cần hiểu nén gây bias hay tăng variance.

### Adaptive Federated Optimization (Reddi et al., 2020 → ICLR 2021)
- Khung tổng quát: **ClientOpt** (SGD cục bộ) + **ServerOpt** coi trung bình cập nhật như một “pseudo-gradient”.
- Dùng Adagrad/Adam/Yogi ở server → **FedAdagrad, FedAdam, FedYogi**; FedAvgM (momentum) là trường hợp riêng.
- Ý quan trọng: tối ưu ở client khác bản chất tối ưu ở server (client chạy ít bước, có thể chỉ tham gia một lần; server giữ được trạng thái).
- Phải tune kỹ cả client LR và server LR.

### Công cụ: TensorFlow Federated (TFF)
Ngôn ngữ khai báo, có kiểu, phân biệt vị trí `@SERVER` / `@CLIENTS`, không đánh chỉ số từng client; có runtime mô phỏng.

### Best practices khi đánh giá thuật toán FL (rất đáng áp dụng)
1. Nói rõ thiết lập: cross-silo hay cross-device (stateful hay stateless).
2. Nói rõ tính toán nào xảy ra ở đâu, gửi gì đi (liên quan hiệu năng và privacy).
3. Hyperparameter tuning quan trọng; ưu tiên thuật toán “tự tune”.
4. Bảo đảm tái lập (reproducibility).
5. Ưu tiên dataset **non-IID tự nhiên**; chia nhãn bệnh lý (pathological label split) không thực tế. Tự hỏi: có thực sự cần một mô hình toàn cục duy nhất?
6. Chọn trục đo cẩn thận. Trục x: số vòng, số byte, thời gian mô phỏng theo chi phí thực (không phải wall-clock mô phỏng). Trục y: accuracy/metric trên **tập test giữ lại** (không phải training loss).
7. Ngoài accuracy/hiệu quả còn có privacy, robustness, fairness, personalization.

Benchmark gợi ý: LEAF (EMNIST, Shakespeare…), Stack Overflow, CIFAR-100 trong TFF.

> **Liên hệ lộ trình:** FedAvg, FedProx, FedAdagrad/FedAdam/FedYogi đều nằm trong baseline bắt buộc ở mục 2.1. Best practices 1–6 nên đưa trực tiếp vào protocol benchmark v1 (Phase 2) và mục 8.5 Reproducibility. Giả định B-dissimilarity của FedProx là một mẫu tham khảo tốt cho phần lý thuyết Paper 2 (bounded client drift / gradient dissimilarity, mục 1.5).

---

## Phần III – Privacy cho FL và Federated Analytics

Mục tiêu: đẩy đường biên Pareto privacy–utility ra xa bằng công nghệ, với ít công sức hơn.

### Ai có thể học được gì?
Mối đe dọa đến từ: mạng, thiết bị, server, mô hình/metric được công bố, mô hình triển khai. Mỗi điểm có công cụ bảo vệ riêng:

| Đối tượng có quyền truy cập | Công cụ |
|---|---|
| Mạng, thiết bị | Mã hóa khi lưu và khi truyền |
| Server (“cách tính”) | Local DP, Secure Enclaves (TEE), MPC, **Secure Aggregation** |
| Mô hình/metric công bố, mô hình triển khai (“tính ra cái gì”) | **Central DP**, kiểm toán privacy thực nghiệm (secret sharer, **membership inference**) |

### Differential Privacy (DP)
(ε, δ)-DP: phân phối đầu ra M(D) và M(D′) gần như nhau với mọi D, D′ kề nhau. Trong FL dùng **user-level DP** (D và D′ khác nhau toàn bộ dữ liệu của một người dùng), mạnh hơn record-level DP.

Quy trình DP-FedAvg: (1) chọn ngẫu nhiên client, (2) **clip** cập nhật theo chuẩn L2 ≤ S, (3) lấy trung bình, (4) **thêm nhiễu** Gauss tỉ lệ với S, (5) cập nhật mô hình. Ví dụ: mô hình ngôn ngữ LSTM trên Reddit đạt (4.63, 1e-9)-DP với 763k user; càng nhiều user, ε càng nhỏ.

- **Central DP:** dễ đạt utility cao nhưng phải tin server.
- **Local DP:** thêm nhiễu ngay trên thiết bị, giả định tin cậy yếu hơn nhưng utility thấp hơn.
- **Distributed DP:** mỗi thiết bị thêm ít nhiễu + **Secure Aggregation** (Bonawitz et al., CCS 2017; Bell et al., CCS 2020) → server chỉ thấy tổng, đạt mức gần central DP (ví dụ ε_local = 1 nhưng ε_central = 0.1).

Secure Aggregation chịu được ~1/3 client độc hại và ~1/3 client rớt; overhead truyền thông ~1.7–2x.

### Thách thức DP trong cross-device thực tế
Không có tập dữ liệu cố định, client online/offline động, có thể rớt giữa chừng → cần cơ chế như **Random Check-ins** để vẫn tính được (ε, δ) chính xác.

> **Liên hệ lộ trình:** Khớp mục 1.6: FL **không tự động** bảo đảm privacy. Nếu luận án đánh giá DP/SecAgg như lớp bổ sung, quy trình clip + noise ở trên là điểm bắt đầu; membership inference là cách kiểm toán thực nghiệm. Lưu ý: cross-silo y tế có ít client (vài site) nên DP mức user/site sẽ tốn utility nhiều hơn ví dụ hàng trăm nghìn user ở trên.

---

## Phần IV – Bài toán mở và chủ đề khác

### 1. Robustness trước lỗi và tấn công
- **Debug khi không xem được dữ liệu:** dùng GAN huấn luyện liên kết để sinh mẫu tổng hợp, ví dụ phát hiện 50% thiết bị có ảnh bị đảo màu.
- **Data poisoning** (rẻ, ảnh hưởng hạn chế) vs **model poisoning** (đắt, gửi gradient tùy ý nên phá hoại mạnh).
- **Untargeted (Byzantine)**: làm giảm hiệu năng chung. **Targeted (backdoor)**: cài hành vi ẩn mà vẫn giữ accuracy (Bagdasaryan et al., AISTATS 2020).
- Phòng thủ đơn giản hiệu quả: **norm bounding** (giới hạn chuẩn cập nhật) + thêm nhiễu.
- Bài toán mở: kẻ tấn công thích nghi, robustness có chứng nhận, đánh đổi giữa privacy và robustness, thiếu benchmark thực tế. “Tấn công phần đuôi phân phối dễ hơn phần đầu.”

### 2. Dữ liệu không đồng nhất: fairness và personalization
- **Fairness:** q-FFL (Li et al., ICLR 2020) nâng loss của client lên lũy thừa q+1 để ưu tiên client đang kém → giảm một nửa phương sai accuracy giữa client mà giữ accuracy trung bình. Liên quan: Agnostic FL (minimax).
- **Personalization**, ba hướng:
  - **Multi-task learning:** mỗi client là một task, học chung nhưng cá nhân hóa (MOCHA, Smith et al., NeurIPS 2017).
  - **Fine-tuning:** học global rồi tinh chỉnh cục bộ (transfer learning, domain adaptation).
  - **Meta-learning:** học điểm khởi tạo tốt rồi huấn luyện cục bộ (Jiang et al. 2019, kiểu MAML; Khodak et al., NeurIPS 2019).

### 3. Thách thức hệ thống
- **Cross-device:** hàng tỷ thiết bị, chỉ một phần online, ≥5% rớt; tương thích phiên bản runtime; không ảnh hưởng pin/mạng; dịch vụ đa người dùng; kích thước quần thể thay đổi (dev ≤12, dogfood ≤1000, production >10 triệu thiết bị).
- **Cross-silo:** nhiều thứ dễ hơn (tin cậy cao, client tham gia mọi vòng, mạng nhanh) nhưng phát sinh: **schema dữ liệu khác nhau** (đặc trưng, nhãn, định dạng), join cho dữ liệu phân hoạch dọc, triển khai phần mềm phức tạp hơn.

> **Liên hệ lộ trình:**
> - Robust aggregation (Median, Trimmed Mean, Krum) ở mục 2.2 xuất phát từ nhánh Byzantine ở đây; lộ trình dùng chúng để **phân biệt** statistical heterogeneity với malicious clients, đúng tinh thần tutorial.
> - Fairness q-FFL là một baseline đáng cân nhắc cho Paper 1 (fairness giữa site lớn/nhỏ, mục 2.2).
> - Ba hướng personalization tương ứng với baseline Paper 3: fine-tuning ↔ FedBABU/local fine-tuning; MTL ↔ Ditto/pFedMe; meta-learning ↔ Per-FedAvg.
> - Thách thức cross-silo về schema khác nhau rất sát với MIA đa trung tâm (máy chụp, protocol khác nhau), là nguồn statistical heterogeneity Paper 1 cần mô tả.

---

## Những điểm cần nhớ

1. FL = dữ liệu ở lại client, chỉ gửi cập nhật để tổng hợp ngay. **Cross-silo ≠ cross-device**: luôn nói rõ mình ở thiết lập nào.
2. FedAvg giảm truyền thông nhờ local updating, nhưng heterogeneity gây drift; FedProx, SCAFFOLD, FedOpt là các cách khắc phục.
3. Đánh giá phải dùng dữ liệu non-IID tự nhiên, trục x theo vòng/byte, trục y theo test accuracy, và có tái lập.
4. FL chưa đủ để bảo đảm privacy; cần DP (user-level) và/hoặc Secure Aggregation.
5. Bài toán mở lớn: robustness, fairness, personalization, hệ thống.

## Đọc tiếp (theo thứ tự ưu tiên cho lộ trình)
1. Kairouz et al., *Advances and Open Problems in FL* (2019/2021), chương 3 (efficiency), 4 (privacy), 6 (fairness).
2. McMahan et al., *Communication-Efficient Learning…* (AISTATS 2017) – FedAvg.
3. Li et al., *Federated Optimization in Heterogeneous Networks* (MLSys 2020) – FedProx.
4. Reddi et al., *Adaptive Federated Optimization* (ICLR 2021) – FedOpt.
5. Li et al., *Fair Resource Allocation in FL* (ICLR 2020) – q-FFL.
6. Smith et al., *Federated Multi-Task Learning* (NeurIPS 2017) – MOCHA.
7. Bonawitz et al., *Practical Secure Aggregation* (CCS 2017); McMahan et al., *Learning DP Recurrent Language Models* (ICLR 2018).
