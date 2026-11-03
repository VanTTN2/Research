# LỘ TRÌNH NGHIÊN CỨU TIẾN SĨ 3 NĂM

## Federated Learning cho môi trường dữ liệu và hệ thống không đồng nhất

**Đơn vị nghiên cứu:** AiTA Lab  
**Thời lượng dự kiến:** 36 tháng  
**Chuyên môn cốt lõi:** Federated Learning (FL)  
**Miền ứng dụng đầu tiên:** Medical Image Analysis (MIA)  
**Nền tảng thực nghiệm bổ sung:** NVIDIA Jetson Orin và Jetson Nano

**Định hướng luận án đề xuất:**

> **Heterogeneity-Aware Adaptive and Personalized Federated Learning**
>
> *Methods and Validation in Cross-Silo Medical Imaging and Heterogeneous Edge Environments*

Tên luận án có thể điều chỉnh sau khi xác định chính xác đóng góp thuật toán ở cuối năm 1. Tránh dùng cụm “Adaptive Federated Optimization” làm tên riêng cho đóng góp mới vì đây đã là tên công trình Reddi et al. (ICLR 2021), đề xuất FedAdagrad, FedAdam và FedYogi.

---

# 1. Định vị khoa học và logic luận án

## 1.1. Chuyên môn và phạm vi

FL là chuyên môn nghiên cứu chính. MIA là miền ứng dụng đầu tiên để kiểm chứng các phương pháp, không phải giới hạn duy nhất của luận án. Luận án tập trung vào hai dạng không đồng nhất cần phân biệt:

1. **Statistical heterogeneity:** dữ liệu giữa các tổ chức/client khác nhau về label prevalence, số lượng mẫu, đặc trưng, domain hoặc phân phối có điều kiện.
2. **System heterogeneity:** client khác nhau về năng lực tính toán, bộ nhớ, thời gian huấn luyện, năng lượng và kết nối mạng.

Hai dạng này có thể cùng xuất hiện nhưng không được đánh đồng. Cross-silo MIA đánh giá chủ yếu statistical heterogeneity giữa các site; Jetson Orin/Nano là testbed để đánh giá system heterogeneity ở tầng edge. Jetson không mặc nhiên đại diện cho các bệnh viện.

## 1.2. Câu hỏi nghiên cứu tổng quát

> How can federated learning achieve stable global optimization and effective client-level personalization under statistical and system heterogeneity, while controlling communication and computation costs?

## 1.3. Chuỗi đóng góp khoa học

Luận án được thiết kế thành một chuỗi ba paper liên tục, trong đó kết quả của paper trước tạo cơ sở cho paper sau:

| Paper | Vấn đề | Đóng góp dự kiến | Vai trò trong luận án |
|---|---|---|---|
| **Paper 1 – Characterization and aggregation** | Cross-silo FL có ít client/site, dữ liệu khác biệt; tất cả site thường tham gia mỗi vòng | Phân tích heterogeneity và client drift; đề xuất/đánh giá cơ chế heterogeneity-aware client weighting hoặc aggregation | Xác định tín hiệu, chỉ số và giới hạn của aggregation hiện tại; tạo đầu vào cho Paper 2 |
| **Paper 2 – Drift-aware adaptive aggregation** | Trọng số cố định hoặc aggregation không phản ánh thay đổi drift theo vòng và theo site | Thuật toán aggregation thích nghi theo tín hiệu drift/heterogeneity; phân tích hội tụ hoặc bảo đảm lý thuyết phù hợp | Đóng góp thuật toán chính về global optimization |
| **Paper 3 – Personalized FL** | Một global model có thể không tối ưu cho mọi site khi khác biệt phân phối lớn | Cơ chế cá nhân hóa có điều kiện dựa trên global representation và thông tin heterogeneity/site | Đóng góp thứ hai về personalized learning, kế thừa Paper 1–2 |

Không nên định nghĩa trước phương pháp cụ thể của Paper 1 khi chưa có literature review và kết quả benchmark. Cần chứng minh khoảng trống so với weighted aggregation, robust aggregation, adaptive optimization và các phương pháp personalized/clustered FL gần nhất.

## 1.4. Định vị cross-silo và cross-device

- **Thiết lập chính cho MIA:** cross-silo FL. Số client ít và tương đối ổn định; thường các site đều tham gia mỗi vòng. FLamby mô tả cross-silo healthcare trong khoảng 2–50 client, tùy dataset và bài toán.
- **Hệ quả:** client selection không phải research question chính của Paper 1 trong cross-silo. Trọng tâm phù hợp hơn là client weighting, update weighting, drift-aware aggregation, robust aggregation và personalization.
- **Cross-device:** có thể được dùng như một thiết lập phụ để nghiên cứu partial participation hoặc resource-aware scheduling, nhưng phải ghi rõ đây là cross-device/simulated setting; không dùng kết quả 50–100 client mô phỏng để đại diện cho 3–10 bệnh viện.
- Các công trình Power-of-Choice, Oort và Clustered Sampling được giữ trong related work nếu cần thảo luận client selection, không phải baseline trung tâm của Paper 1 cross-silo.

## 1.5. Định vị về lý thuyết và hội tụ

Luận án ưu tiên **algorithmic FL research có kiểm chứng thực nghiệm**, không tuyên bố là luận án tối ưu hóa lý thuyết thuần túy. Tuy nhiên, Paper 2 nên có ít nhất một phân tích lý thuyết có ý nghĩa, chẳng hạn convergence bound hoặc proposition dưới các giả định chuẩn:

- smoothness của objective;
- bounded stochastic gradient variance;
- bounded client drift / gradient dissimilarity;
- điều kiện learning rate và số local steps;
- full participation cho phân tích cross-silo cơ bản; mở rộng partial participation nếu thực sự cần.

Nếu không thể chứng minh hội tụ đầy đủ cho mô hình phi lồi, cần đưa ra kết quả yếu hơn nhưng chính xác (ví dụ bound cho expected squared gradient norm hoặc convergence to a stationary point), nêu rõ assumptions và giới hạn. Không dùng cụm “convergence guarantee” nếu chỉ có đồ thị loss/accuracy giảm theo round.

## 1.6. Phạm vi privacy và security

FL không tự động bảo đảm privacy. Luận án cần phân biệt rõ:

- **Phạm vi chính:** tối ưu hóa và cá nhân hóa dưới statistical/system heterogeneity.
- **Privacy baseline:** mô tả threat model, dữ liệu không rời site, kiểm soát truy cập và hạn chế của việc chia sẻ gradients/updates.
- **Secure aggregation / Differential Privacy (DP):** khảo sát và có thể đánh giá overhead như một lớp bảo vệ bổ sung nếu thời gian cho phép; không tuyên bố DP hoặc secure aggregation là đóng góp chính nếu không có thiết kế, phân tích và đánh giá riêng.
- Kết quả về robustness trước malicious clients từ KMFed của AiTA Lab không đồng nghĩa với bảo đảm privacy.

---

# 2. Các trụ cột nghiên cứu và baseline

## 2.1. Federated optimization và drift

Bắt buộc nắm vững và triển khai/đối chiếu:

- FedAvg; FedProx; SCAFFOLD; FedNova.
- FedAdagrad, FedAdam, FedYogi (FedOpt / adaptive server optimization).
- FedDyn (dynamic regularization).
- MOON (model-contrastive learning).
- FedDC – **Gao et al., CVPR 2022, “FedDC: Federated Learning With Non-IID Data via Local Drift Decoupling and Correction,” DOI: 10.1109/CVPR52688.2022.00987.** Luôn ghi tác giả/năm/tên đầy đủ vì acronym FedDC có thể bị trùng trong các tài liệu khác.

## 2.2. Heterogeneity-aware weighting / aggregation

Cần khảo sát ít nhất:

- Sample-size weighting (weighted FedAvg) và uniform averaging.
- Update-norm / update-similarity based weighting.
- Drift-aware weighting và control-variate methods.
- Robust aggregation: coordinate-wise Median, Trimmed Mean, Krum/Multi-Krum, FLTrust (để phân biệt statistical heterogeneity với malicious-client robustness).
- Cluster-aware aggregation nếu có bằng chứng clients thuộc các nhóm phân phối khác nhau.

**Nguyên tắc:** trọng số theo số mẫu không đồng nghĩa với trọng số theo độ tin cậy hoặc mức độ đại diện. Một phương pháp weighting phải phân tích được bias–variance trade-off, fairness giữa site lớn/nhỏ và nguy cơ loại bỏ site hiếm nhưng có ý nghĩa lâm sàng.

## 2.3. Personalized FL

Baseline cần đưa vào mục tiêu nghiên cứu và benchmark, không chỉ nhắc ở phần tổng quan:

- Ditto; pFedMe; FedPer; FedRep.
- APFL (Adaptive Personalized Federated Learning).
- Per-FedAvg (meta-learning based personalization).
- FedBABU (shared representation learning, local head fine-tuning).
- Local fine-tuning và centralized/local-only models.
- IFCA và Clustered Federated Learning (CFL) nếu hướng nghiên cứu dùng clustering hoặc client groups.

Chọn baseline theo kiến trúc và bài toán cụ thể; không nhất thiết chạy mọi phương pháp trên mọi dataset nếu chi phí triển khai không tương xứng.

## 2.4. Client selection – phạm vi phụ

Power-of-Choice, Oort và Clustered Sampling là các công trình quan trọng cho cross-device/partial participation. Chỉ đưa vào thực nghiệm khi có câu hỏi riêng về partial participation, client availability hoặc resource-aware scheduling. Với cross-silo MIA, thiết lập chính nên là tất cả site tham gia mỗi vòng; không biến client selection thành đóng góp cốt lõi nếu không có bối cảnh thực tế hỗ trợ.

---

# 3. Lộ trình 36 tháng đã điều chỉnh

| Giai đoạn | Thời gian | Trọng tâm | Deliverable / Gate |
|---|---:|---|---|
| Phase 1 | M1–M2 | Đánh giá đầu vào; bổ sung ML, optimization, PyTorch song song với FL reading | Diagnostic report; code ML/PyTorch; kế hoạch bù kiến thức cá nhân |
| Phase 2 | M2–M5 | FL fundamentals; triển khai baseline; đọc cross-silo và cross-device | FL benchmark v1; baseline reproducible |
| Phase 3 | M4–M9 | MIA/FLamby; phân tích heterogeneity; chốt RQ Paper 1; thực nghiệm và viết bài | Paper 1 submitted muộn nhất M8–M9 |
| Phase 4 | M10–M18 | Drift-aware Adaptive Aggregation; lý thuyết và thực nghiệm | Paper 2 submitted trước M18 |
| Phase 5 | M17–M27 | Personalized FL kế thừa tín hiệu từ Paper 1–2 | Methodology + Paper 3 submitted khoảng M25–M27 |
| Phase 6 | M22–M30 | Edge FL trên Jetson Orin/Nano; đánh giá system heterogeneity; chỉ chọn một nhánh mở rộng | Edge testbed và kết quả kiểm chứng |
| Phase 7 | M28–M36 | Tích hợp kết quả, validation MIA, viết luận án, phản biện và bảo vệ | Thesis hoàn chỉnh; hồ sơ công bố theo quy định |

Các phase có thể chồng lấn. Mốc “submitted”, “under review”, “accepted” và “published” phải được báo cáo riêng. Không đặt acceptance ở M12 làm điều kiện chắc chắn vì thời gian phản biện phụ thuộc venue.

---

# 4. Năm 1 – Nền tảng FL, cross-silo benchmark và Paper 1

## Phase 1: Diagnostic và nền tảng (M1–M2)

### Nội dung

- Đánh giá đầu vào về linear algebra, probability/statistics, calculus, optimization, ML, Python và PyTorch.
- Bổ sung theo lỗ hổng thực tế thay vì dành cố định bốn tháng cho toàn bộ ML cơ bản.
- Học song song: mỗi tuần đọc ít nhất một bài FL nền tảng và triển khai một thí nghiệm nhỏ.
- Nắm Logistic Regression, MLP/CNN cơ bản, SGD, momentum, Adam, regularization, loss và evaluation metrics.

### Deliverable M2

- Diagnostic report và individual learning plan.
- PyTorch training pipeline có seed, config và log.
- Giải thích được empirical risk, gradient, SGD, local/global objective.

## Phase 2: FL fundamentals và benchmark (M2–M5)

### Kiến thức

- Centralized, distributed và federated learning.
- Cross-silo vs cross-device; synchronous vs asynchronous FL.
- FedAvg, FedProx, SCAFFOLD, FedNova, FedAdam/FedYogi/FedAdagrad.
- Statistical heterogeneity, system heterogeneity, client drift.
- Privacy/security fundamentals và threat model.

### Objective function

```latex
\min_{w} F(w) = \sum_{k=1}^{K} p_k F_k(w), \quad
F_k(w) = \frac{1}{n_k}\sum_{i=1}^{n_k} \ell(w; x_{k,i}, y_{k,i}), \quad
p_k = \frac{n_k}{\sum_{j} n_j}
```

Trong đó:

- `K`: số client/site.
- `n_k`: số mẫu tại client `k`.
- `p_k`: trọng số theo sample size trong weighted FedAvg.
- `F_k(w)`: local empirical objective.

### Thực nghiệm benchmark v1

1. Centralized vs local-only vs FedAvg.
2. IID vs các kiểu Non-IID có kiểm soát.
3. FedAvg vs FedProx vs SCAFFOLD vs FedNova.
4. Bổ sung FedAdam, FedYogi, FedAdagrad; FedDyn, MOON và FedDC theo task phù hợp.
5. Full participation là thiết lập mặc định cho cross-silo; partial participation chỉ là thí nghiệm bổ sung.
6. Đo global performance, per-client performance, worst-client performance, convergence, communication và runtime.

Không dùng số lượng 50–100 clients làm cấu hình MIA cross-silo mặc định. Với benchmark mô phỏng tổng quát có thể dùng nhiều client, nhưng phải ghi nhãn cross-device/synthetic-client setting.

### Deliverable M5

- FL benchmark v1 có khả năng tái lập.
- Báo cáo baseline và literature matrix.
- Bộ script tạo partition IID/Non-IID; log và config đầy đủ.

## Phase 3: Cross-silo MIA và Paper 1 (M4–M9)

### Research focus

**Đề tài làm việc:**

> *Heterogeneity-Aware Client Weighting and Aggregation for Cross-Silo Federated Medical Image Learning*

Đây là working title, cần điều chỉnh sau khi xác định novelty so với literature.

### Research questions dự kiến

- RQ1: Các dạng statistical heterogeneity giữa site ảnh hưởng thế nào đến global performance, client-level performance và drift?
- RQ2: Các chỉ số đo heterogeneity/drift nào có thể dùng để điều chỉnh client contribution mà không làm giảm khả năng học của site nhỏ hoặc site có phân phối hiếm?
- RQ3: Heterogeneity-aware weighting/aggregation cải thiện trade-off giữa global performance, worst-site performance và convergence so với weighted FedAvg, FedProx, SCAFFOLD và FedOpt như thế nào?

### Experimental design

- Bối cảnh chính: cross-silo, số site theo split tự nhiên của dataset; tất cả site tham gia mỗi vòng trong thiết lập chính.
- Baseline: uniform averaging, sample-size weighted FedAvg, FedProx, SCAFFOLD, FedNova; FedAdam/FedYogi; robust aggregation khi có lý do.
- Phân tích: client update norm, cosine similarity, gradient/update drift, local validation performance, site size và label/domain shift.
- Không dùng client selection làm giải pháp chính cho cross-silo. Nếu thêm partial participation, phải tách thành experiment phụ và giải thích kịch bản vận hành.
- Đánh giá nhiều seeds, confidence intervals, ablation và sensitivity analysis.

### Mốc Paper 1

| Thời điểm | Công việc |
|---|---|
| M4–M5 | Literature matrix; chọn dataset; tái lập FedAvg và baseline đầu tiên |
| M6 | Chốt research question, hypothesis, protocol và venue shortlist |
| M6–M7 | Hoàn tất benchmark chính; xác định novelty và baseline gần nhất |
| M7–M8 | Viết bản thảo; internal review tại AiTA Lab |
| **Muộn nhất M8–M9** | **Submit Paper 1** |
| M10–M15 | Theo dõi review, rebuttal/revision hoặc chuyển venue nếu cần |

**Mục tiêu thực tế:** Paper 1 được submit trong năm 1; acceptance trong năm 1 là stretch goal phụ thuộc lịch venue, không phải cam kết. Nếu quy định chương trình yêu cầu bài accepted/published, phải lập publication buffer và kiểm tra điều kiện tốt nghiệp ngay từ đầu.

---

# 5. Năm 2 – Drift-aware Adaptive Aggregation và Personalized FL

## Phase 4: Drift-aware Adaptive Aggregation (M10–M18)

### Research theme

> **Drift-aware Adaptive Aggregation for Cross-Silo Federated Learning**

Tên này mô tả vấn đề và tránh nhầm với paper “Adaptive Federated Optimization” của Reddi et al. (ICLR 2021).

### Research hypothesis

Các tín hiệu drift/heterogeneity được quan sát qua client updates và lịch sử huấn luyện có thể dùng để điều chỉnh mức đóng góp của từng site theo thời gian, cải thiện ổn định và client-level performance so với aggregation cố định, trong khi kiểm soát chi phí tính toán/truyền thông.

### Các bước

1. Phân tích kết quả Paper 1 để chọn tín hiệu có giá trị và tránh tái sử dụng heuristic không có cơ sở.
2. Xây dựng formulation và xác định rõ weighting theo sample size, reliability, drift hay utility khác nhau thế nào.
3. Thiết kế thuật toán thích nghi theo round; phân tích độ phức tạp và chi phí truyền thông.
4. Thực nghiệm cross-silo full participation; thêm partial participation chỉ khi có giả thuyết cụ thể.
5. So sánh baseline: FedAvg, FedProx, SCAFFOLD, FedNova, FedAdam, FedYogi, FedAdagrad, FedDyn, MOON và FedDC (Gao et al., CVPR 2022).
6. Thực hiện ablation, sensitivity, robustness với mức heterogeneity khác nhau và đánh giá worst-site.
7. Hoàn thiện phân tích hội tụ dưới assumptions phù hợp; nêu rõ giới hạn lý thuyết.

### Phân tích lý thuyết tối thiểu cần hướng tới

- Định nghĩa global objective và local objectives.
- Nêu assumptions (smoothness, bounded variance, bounded heterogeneity/drift, learning-rate constraints).
- Phân tích sai lệch giữa aggregated update và global gradient.
- Chứng minh bound cho expected gradient norm hoặc stationary-point convergence trong setting đã định nghĩa.
- Phân tích ảnh hưởng của adaptive weights đến bound và các điều kiện để weights không gây bias không kiểm soát.

Không cần ép một theorem tổng quát cho mọi deep network và mọi Non-IID setting. Một kết quả lý thuyết có phạm vi hẹp nhưng đúng, rõ assumptions và được thực nghiệm kiểm chứng có giá trị hơn một phát biểu hội tụ mơ hồ.

### Mốc và deliverable

- M10–M12: formulation, proof plan, baseline reproduction.
- M13–M15: thuật toán và thực nghiệm chính.
- M16–M17: convergence analysis, ablation, manuscript.
- **Trước M18: submit Paper 2.**

Paper 2 cần submit trước M18 để tạo đủ thời gian phản biện/revision. Không xem “manuscript hoàn thiện” là tương đương “đã submit”.

## Phase 5: Personalized Federated Learning (M17–M27)

### Research theme

> **Heterogeneity-Guided Personalized Federated Learning**

### Logic kế thừa

- Paper 1 xác định mức và kiểu heterogeneity giữa site.
- Paper 2 cung cấp global model/aggregation và tín hiệu drift đã được điều chỉnh.
- Paper 3 dùng các thông tin này để quyết định phần nào nên chia sẻ, phần nào nên cá nhân hóa; không giả định global model tốt hơn đồng nghĩa với personalized model tốt hơn.

### Research questions

1. Khi nào một global representation có lợi cho mọi site, và khi nào cần local adaptation?
2. Có thể sử dụng heterogeneity/drift indicators từ Paper 1–2 để điều chỉnh mức shared-private learning không?
3. Phương pháp có cải thiện average client performance, worst-client performance và generalization tới client/site mới không?

### Baselines

- Ditto, pFedMe, FedPer, FedRep.
- APFL, Per-FedAvg, FedBABU.
- Local fine-tuning; global-only FedAvg/FedOpt.
- IFCA/CFL nếu đề xuất có clustering hoặc client grouping.

### Mốc và deliverable

- M17–M19: literature review và xác định khác biệt với Paper 2.
- M20–M22: formulation, prototype và baseline.
- M23–M25: thực nghiệm, ablation và đánh giá site-level.
- **M25–M27: submit Paper 3.**

Không đặt mục tiêu acceptance Paper 3 trước khi có đủ kết quả review. Cần tính đến thời gian journal review 6–12 tháng hoặc lâu hơn, tùy tạp chí.

---

# 6. Năm 3 – Edge FL, validation và tích hợp luận án

## Phase 6: Heterogeneous Edge FL testbed (M22–M30)

AiTA Lab có Jetson Orin và Jetson Nano để kiểm chứng system heterogeneity. Đây là một testbed bổ sung cho FL algorithm, không thay thế cross-silo MIA.

### Kiến trúc

- FL server: GPU server của AiTA Lab (ví dụ RTX 3090).
- Edge clients: Jetson Orin và Jetson Nano; mỗi thiết bị chạy một client process hoặc nhiều client logic khi phù hợp.
- Orchestration: Flower hoặc framework tương thích; PyTorch; Docker nếu khả thi trên từng nền tảng.
- Network: LAN ở giai đoạn đầu; sau đó có thể mô phỏng bandwidth/latency nếu cần.

### Câu hỏi thực nghiệm

- Mức chênh lệch về local training time, memory, throughput và communication ảnh hưởng thế nào đến synchronous FL?
- Điều chỉnh batch size/local epochs/model size theo tài nguyên thiết bị có cải thiện hiệu quả thực thi không?
- Adaptive scheduling/aggregation có giảm straggler effect mà không làm lệch đóng góp dữ liệu không?

### Metrics

- Local training time và round time.
- Peak RAM, CPU/GPU utilization.
- Model size và bytes transmitted.
- Throughput và inference latency.
- Energy/power nếu có thiết bị đo đáng tin cậy.
- Accuracy/F1 và per-client performance.

### Nguyên tắc thực nghiệm

- Jetson Orin là thiết bị chính cho local training; Jetson Nano, nhất là bản 2GB, dùng mô hình nhỏ, batch size thấp, lightweight training hoặc inference tùy giới hạn thực tế.
- Ghi rõ model, RAM, JetPack/CUDA/PyTorch versions, power mode và nhiệt độ/thermal throttling nếu có ảnh hưởng.
- Không so sánh công bằng bằng cách ép mọi thiết bị dùng cùng batch size nếu điều đó làm một thiết bị không thể chạy; phải công bố cấu hình và chuẩn hóa metric hợp lý.
- Không gọi đây là “hospital cross-silo FL”. Đây là edge/system-heterogeneous FL testbed.

### Nhánh mở rộng

Chỉ chọn một: communication-efficient FL, resource-aware FL hoặc federated active learning. Không dùng acronym “FDAL” trong roadmap khi chưa ghi rõ tên đầy đủ và bài báo tương ứng. Nếu kế thừa một công trình Active Learning của AiTA Lab, cần dẫn citation chính xác và phân biệt với Federated Active Learning (FedAL).

## Phase 7: MIA validation và thesis integration (M28–M36)

- Chọn một task MIA chính: classification hoặc segmentation; không mở rộng đồng thời quá nhiều modality.
- Đánh giá cùng protocol cho Paper 1–3; bảo đảm kết quả so sánh được.
- Nếu có điều kiện, thêm một ứng dụng ngoài MIA (ví dụ SER) chỉ để kiểm tra tính tổng quát; không làm thành trụ cột thứ ba.
- M28–M31: hoàn thiện thí nghiệm và manuscript revision.
- M31–M34: tích hợp luận án, kiểm tra consistency giữa các paper.
- M34–M36: hoàn thiện hồ sơ, phản biện nội bộ và bảo vệ theo quy định chương trình.

---

# 7. Publication roadmap và quản trị rủi ro thời gian

| Paper | Nội dung | Mốc submit mục tiêu | Mốc acceptance |
|---|---|---:|---|
| Paper 1 – Conference | Cross-silo heterogeneity characterization + weighting/aggregation; MIA | M8–M9 | Không cam kết M12; phụ thuộc venue/review |
| Paper 2 – Journal | Drift-aware Adaptive Aggregation + convergence analysis | Trước M18 | Có thể sau M24 tùy journal/revision |
| Paper 3 – Journal | Heterogeneity-guided Personalized FL | M25–M27 | Có rủi ro sau M36; cần chọn journal và submit sớm nhất có thể |
| Paper 4 – Optional | Edge FL / MIA extension / SER validation | Chỉ khi Paper 1–3 không bị ảnh hưởng | Không phải điều kiện cốt lõi |

**Lưu ý tiến độ tiến sĩ:** cần kiểm tra chính thức quy định của chương trình về bài accepted, online first, published, quartile/indexing, thời điểm chốt hồ sơ và yêu cầu tác giả. Không dùng manuscript hoặc submission để thay thế tiêu chí accepted/published nếu quy chế không cho phép.

**Publication buffer:** nếu Paper 3 phải được accepted trước khi nộp luận án, cần dời mốc submit lên sớm hơn (lý tưởng M20–M22) hoặc chọn chiến lược công bố khác sau khi kiểm tra quy định. Đây là quyết định cần chốt ở M12, không đợi đến năm 3.

---

# 8. Datasets và experimental protocol cho MIA

## 8.1. Ưu tiên FLamby cho cross-silo healthcare

Bổ sung **FLamby (NeurIPS 2022 Datasets and Benchmarks)** làm nguồn benchmark ưu tiên vì bộ này được thiết kế cho cross-silo FL trong healthcare với natural/site splits. Các dataset cần khảo sát gồm:

- **Fed-ISIC2019:** skin lesion classification, 6 centers.
- **Fed-IXITiny:** brain MRI (T1), 3 centers.
- **Fed-KITS19:** kidney CT segmentation; 6 silos được tạo từ metadata của 87 centers, chỉ giữ các silo có trên 10 samples theo mô tả của repository. License được ghi là CC BY-NC-SA 4.0; cần tuân thủ license và điều khoản gốc.
- **Fed-LIDC-IDRI:** lung CT dataset, 2 centers, phù hợp khảo sát classification theo benchmark FLamby.
- **Fed-Camelyon16:** histopathology classification, 2 centers.

Ưu tiên chọn 1 dataset classification (Fed-ISIC2019 hoặc Fed-Camelyon16) cho Paper 1, sau đó dùng Fed-KITS19 hoặc Fed-IXITiny nếu muốn mở rộng sang segmentation/3D imaging. FLamby có nhiều dataset và task; không mặc định tất cả phù hợp cho cùng một kiến trúc hoặc cùng mức độ truy cập. Kiểm tra preprocessing, số site, số mẫu mỗi site, nhãn, license/terms of use và yêu cầu đăng ký trước khi sử dụng.

## 8.2. Dataset bổ sung và giới hạn

| Dataset | Vai trò | Lưu ý |
|---|---|---|
| MedMNIST | Smoke test, prototyping, baseline nhanh | Không coi là cross-silo thực tế nếu partition tự tạo; kiểm tra license từng subset, có subset CC BY-NC |
| NIH ChestX-ray14 | Chest X-ray classification benchmark | Nguồn dữ liệu từ một hệ thống/trung tâm; nhãn được khai thác tự động bằng NLP và có nhiễu; không dùng để tuyên bố multi-hospital FL |
| BraTS | Brain tumor MRI/segmentation | Cần đăng ký/chấp thuận truy cập theo điều kiện challenge; kiểm tra terms từng phiên bản |
| Medical Segmentation Decathlon | Segmentation benchmark | Kiểm tra task-specific license, provenance và tính phù hợp với FL split |
| FLamby (Fed-ISIC2019, Fed-Camelyon16, Fed-IXITiny, Fed-KITS19, Fed-LIDC-IDRI) | Cross-silo site-based evaluation | Ưu tiên natural splits; kiểm tra dataset-specific license và báo cáo đúng site count/provenance |

Nếu dùng dataset một trung tâm như NIH ChestX-ray14, có thể tạo simulated clients để stress-test thuật toán, nhưng phải ghi rõ đây là partition nhân tạo, không phải bệnh viện độc lập.

## 8.3. Non-IID scenarios

- Natural site/domain shift từ FLamby.
- Label prevalence shift.
- Quantity imbalance.
- Feature/domain shift.
- Concept shift nếu có dữ liệu và thiết kế phù hợp.
- Dirichlet α (ví dụ 0.1, 0.3, 0.5, 1.0) chỉ là simulated label-skew setting, không thay thế natural site splits.

## 8.4. Metrics

- Predictive: Accuracy, Macro-F1, AUC, sensitivity/specificity; Dice/IoU cho segmentation.
- Client-level: mean, median, standard deviation, worst-site performance.
- Optimization: rounds to target, expected gradient norm nếu có phân tích lý thuyết, update drift.
- Communication: bytes/round, total bytes.
- Computation: training time, inference latency, memory, GPU/CPU use.
- Fairness/clinical relevance: không chỉ báo cáo global average; phân tích site nhỏ và site có prevalence hiếm.

## 8.5. Reproducibility

Mỗi thí nghiệm phải lưu dataset/version, license/terms, preprocessing, site split, random seed, model, hyperparameters, optimizer, FL config, software versions, checkpoints, logs và evaluation script. Báo cáo mean ± standard deviation hoặc confidence interval phù hợp trên nhiều seeds; với natural site splits có số site ít, cần diễn giải thận trọng và không giả vờ có statistical power lớn.

---

# 9. Kế thừa các công trình và tài nguyên của AiTA Lab

## 9.1. KMFed

Theo thông tin nghiên cứu hiện có của lab, KMFed sử dụng task-utility/anomaly detection dựa trên validation mAP@0.5, 1D K-Means (K=2) và safe-cluster aggregation; đã thử nghiệm trên KITTI và PASCAL VOC 2007 với các baseline như FedAvg, FLTrust, Krum, Median và Trimmed Mean.

**Tài nguyên có thể kế thừa:**

- Pipeline validation-based client scoring và aggregation.
- K-Means filtering / safe-cluster selection logic.
- Attack configurations, Non-IID partitions và experimental scripts.
- Logging, reproducibility setup và code repository (sau khi kiểm tra phiên bản hiện tại).

**Ranh giới khoa học:** KMFed tập trung vào phát hiện/lọc client bất thường hoặc malicious-client robustness. Luận án mới tập trung vào statistical heterogeneity và client drift trong cross-silo FL. Có thể dùng KMFed làm baseline/robustness reference, nhưng không được xem hai bài toán là tương đương hoặc tái sử dụng nguyên phương pháp rồi tuyên bố novelty mới.

## 9.2. FedalMER

FedalMER được mô tả là **“Federated Dynamic Gating Framework for Semi-Supervised Multimodal Emotion Recognition”**, kết hợp FL, semi-supervised learning, audio/text, Dynamic Hypergraph Learning, Gated Cross-Modal Transformer, Gated Multimodal Unit và confidence-based pseudo-labeling; đã khảo sát IEMOCAP và MELD.

**Tài nguyên/kinh nghiệm có thể kế thừa sau khi kiểm tra code và protocol:**

- FL orchestration và pipeline phân phối dữ liệu audio/text.
- Kinh nghiệm multi-client, labeled-ratio settings và aggregation strategies.
- Multimodal model components và pseudo-labeling nếu cần validation ngoài MIA.

**Vai trò trong luận án:** FedalMER là bằng chứng AiTA Lab đã có kinh nghiệm triển khai FL ở một miền ứng dụng khác. Có thể dùng SER làm external validation cuối luận án nếu phương pháp tổng quát hóa tốt; không cần biến SER thành một research pillar mới.

## 9.3. FDAL và Federated Active Learning

Không sử dụng “FDAL” như một acronym chưa định nghĩa. Cần xác minh tên đầy đủ, citation, phạm vi và quan hệ với công trình Active Learning của AiTA Lab trước khi đưa vào related work. Nếu luận án muốn nghiên cứu federated active learning, dùng tên đầy đủ **Federated Active Learning (FedAL)** và xác định rõ đây là nhánh khác với client weighting/aggregation.

---

# 10. Kỹ năng, tổ chức nhóm và milestone

## 10.1. Năng lực cần đạt

| Năng lực | Cuối năm 1 | Cuối năm 2 | Cuối năm 3 |
|---|---|---|---|
| Python/PyTorch | Tự triển khai baseline | Thiết kế module nghiên cứu | Code/repository reproducible |
| ML/Optimization | Nắm nền tảng cần thiết | Formulation và phân tích | Phản biện assumptions/limitations |
| FL | Tái lập baseline | Đề xuất aggregation/personalization | Tổng hợp và khái quát hóa |
| Mathematics | Hiểu objective/gradient | Phân tích hội tụ có phạm vi | Trình bày giới hạn lý thuyết |
| Experimental design | Benchmark chuẩn | Ablation, sensitivity, fairness | Cross-site/edge validation |
| Research writing | Paper 1 submitted | Paper 2 submitted | Paper 3, thesis |

## 10.2. Phân công NCS và sinh viên

- **NCS:** chịu trách nhiệm RQ, formulation, algorithm design, assumptions, novelty, experimental protocol, phân tích kết quả và manuscript chính.
- **Sinh viên:** literature matrix theo chủ đề hẹp, dataset preprocessing, baseline reproduction, test scripts, edge-device profiling và documentation.
- **Giảng viên/mentor AiTA Lab:** review novelty, kiểm tra protocol, hỗ trợ MIA/Edge deployment và phản biện định kỳ.
- Authorship dựa trên đóng góp thực chất; không mặc định đồng tác giả chỉ vì tham gia nhóm.

## 10.3. Milestones và decision gates

| Mốc | Tiêu chí đánh giá |
|---|---|
| M2 | Diagnostic hoàn tất; PyTorch pipeline chạy được |
| M5 | FL benchmark v1; hiểu cross-silo/cross-device và baseline chính |
| M6 | Chốt RQ Paper 1, dataset, protocol và venue shortlist |
| M8–M9 | Paper 1 submitted |
| M12 | Quyết định Paper 1; Paper 2 có formulation/proof plan và baseline |
| M18 | Paper 2 submitted; có kết quả chính và phân tích hội tụ phù hợp |
| M22 | Chốt scope Paper 3; quyết định publication buffer theo quy chế tiến sĩ |
| M25–M27 | Paper 3 submitted |
| M30 | Edge FL validation hoàn tất hoặc dừng nếu không đóng góp vào thesis |
| M36 | Luận án và hồ sơ bảo vệ hoàn chỉnh |

---

# 11. Rủi ro và phương án kiểm soát

| Rủi ro | Biện pháp |
|---|---|
| Nền tảng ML/optimization chưa vững | Diagnostic M1; học theo lỗ hổng; học song song FL và coding |
| Paper 1 thiếu novelty | Chốt RQ M6; benchmark cross-silo thực; so sánh weighting/aggregation gần nhất; không dựa vào client selection giả lập |
| Acceptance conference trễ | Submit M8–M9; theo dõi venue timeline; phân biệt acceptance với submission |
| Paper 2 thiếu lý thuyết | Chọn assumptions và proof target từ M10; review sớm; thu hẹp setting nếu cần |
| Journal review kéo dài | Submit Paper 2 trước M18; Paper 3 trước M27 hoặc sớm hơn theo quy chế |
| Dataset không truy cập được | Kiểm tra FLamby, license và đăng ký ngay M4; có dataset dự phòng nhưng không đánh đồng simulated với natural sites |
| MIA quá rộng | Chọn một task chính; ưu tiên classification trước segmentation nếu phù hợp |
| Privacy bị reviewer chất vấn | Viết threat model và scope rõ; không tuyên bố FL tự bảo đảm privacy; đánh giá secure aggregation/DP nếu claim yêu cầu |
| Edge testbed làm loãng luận án | Xem Jetson là validation system heterogeneity; chỉ giữ nếu hỗ trợ RQ; không thay thế cross-silo MIA |
| Trùng lặp với KMFed/FedalMER | Lập bảng contribution overlap; nêu rõ khác biệt task, threat model, method và datasets |
| Paper 3 không accepted trước khi bảo vệ | Kiểm tra quy chế từ M1; lập publication buffer; ưu tiên submit sớm và chọn venue phù hợp |

---

# 12. Tài liệu nền tảng và baseline cần đọc

## 12.1. Optimization / heterogeneity

1. McMahan et al. (2017), “Communication-Efficient Learning of Deep Networks from Decentralized Data,” AISTATS – FedAvg.
2. Li et al. (2020), “Federated Optimization in Heterogeneous Networks,” MLSys – FedProx.
3. Karimireddy et al. (2020), “SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,” ICML.
4. Wang et al. (2020), “Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization,” NeurIPS – FedNova.
5. Reddi et al. (2021), “Adaptive Federated Optimization,” ICLR – FedAdagrad, FedAdam, FedYogi. DOI/arXiv: 10.48550/arXiv.2003.00295.
6. Acar et al. (2021), “Federated Learning Based on Dynamic Regularization,” ICLR – FedDyn.
7. Li et al. (2021), “Model-Contrastive Federated Learning,” CVPR – MOON.
8. Gao et al. (2022), “FedDC: Federated Learning With Non-IID Data via Local Drift Decoupling and Correction,” CVPR. DOI: 10.1109/CVPR52688.2022.00987.

## 12.2. Client selection – related work, không phải trọng tâm cross-silo

1. Cho et al. (2020/2022), “Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies.”
2. Lai et al. (2021), “Oort: Efficient Federated Learning via Guided Participant Selection,” OSDI.
3. Fraboni et al. (2021), “Clustered Sampling: Low-Variance and Improved Representativity for Clients Selection in Federated Learning,” ICML.

## 12.3. Personalized / clustered FL

1. Fallah et al. (2020), “Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach,” NeurIPS – Per-FedAvg.
2. Li et al. (2020), “Exploiting Shared Representations for Personalized Federated Learning,” ICML – FedRep.
3. Dinh et al. (2020), “Personalized Federated Learning with Moreau Envelopes,” NeurIPS – pFedMe.
4. Li et al. (2021), “Ditto: Fair and Robust Federated Learning Through Personalization,” ICML.
5. Oh et al. (2022), “FedBABU: Toward Enhanced Representation for Federated Image Classification,” ICLR.
6. Ghosh et al. (2020), “An Efficient Framework for Clustered Federated Learning,” NeurIPS – IFCA.
7. Sattler et al. (2021), “Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization Under Privacy Constraints,” IEEE TNNLS – CFL.
8. Deng, Kamani & Mahdavi (2020), “Adaptive Personalized Federated Learning,” arXiv:2003.13461 – APFL.

## 12.4. Healthcare cross-silo benchmark

Ogier du Terrail et al. (2022), “FLamby: Datasets and Benchmarks for Cross-Silo Federated Learning in Realistic Healthcare Settings,” NeurIPS 2022 Datasets and Benchmarks Track. DOI: 10.52202/068431-0384.

## 12.5. Nguồn tra cứu

- FLamby: https://github.com/owkin/FLamby
- FLamby documentation: https://owkin.github.io/FLamby/
- NeurIPS Proceedings: https://papers.nips.cc/
- PMLR: https://proceedings.mlr.press/
- CVF Open Access: https://openaccess.thecvf.com/
- Flower: https://flower.ai/
- PyTorch: https://pytorch.org/

---

# 13. Kết luận và khuyến nghị chính thức

1. **Paper 1 phải phù hợp cross-silo MIA:** ưu tiên heterogeneity-aware client weighting/aggregation, không lấy client selection làm trọng tâm.
2. **Ba paper phải tạo thành một chuỗi:** characterization/aggregation → drift-aware adaptive aggregation → personalized FL.
3. **Paper 2 cần phân biệt với Reddi et al. (2021):** dùng tên Drift-aware Adaptive Aggregation và đưa FedAdam/FedYogi/FedAdagrad vào baseline.
4. **Có mục tiêu phân tích hội tụ rõ ràng:** ít nhất một bound/proposition có assumptions phù hợp; nếu luận án chỉ thực nghiệm thì phải định vị minh bạch và không tuyên bố đóng góp optimization theory.
5. **FLamby là benchmark ưu tiên cho healthcare cross-silo:** NIH ChestX-ray14 và MedMNIST chỉ dùng đúng vai trò, không gán provenance đa bệnh viện giả.
6. **Privacy được định vị đúng phạm vi:** FL không đồng nghĩa với privacy guarantee; secure aggregation/DP cần được mô tả và đánh giá nếu claim đòi hỏi.
7. **Kế thừa KMFed và FedalMER có chọn lọc:** tái sử dụng pipeline, kinh nghiệm và benchmark; đồng thời phân định novelty để tránh trùng lặp.
8. **Jetson Orin/Nano là edge testbed:** hỗ trợ kiểm chứng system heterogeneity, không thay thế cross-silo MIA.
9. **Quản trị publication thực tế:** Paper 1 submit M8–M9; Paper 2 submit trước M18; Paper 3 submit M25–M27 hoặc sớm hơn theo quy chế tiến sĩ. Acceptance/publication không được xem là chắc chắn.

**Định vị cuối cùng:** AiTA Lab phát triển năng lực nghiên cứu FL theo hướng heterogeneity-aware optimization và personalization, có kiểm chứng trên cross-silo healthcare datasets và heterogeneous edge hardware. MIA là ứng dụng mở đầu; phương pháp cần được thiết kế đủ tổng quát để có thể chuyển sang SER, multimodal learning hoặc IoT/Edge AI khi có bằng chứng thực nghiệm.
