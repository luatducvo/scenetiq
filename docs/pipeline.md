# Pipeline của SceneTIQ

SceneTIQ biến một bộ ảnh 2D không có thông tin camera thành đám mây điểm 3D, tách thành các **Instance** để tìm bằng văn bản tự do. Hệ thống có hai luồng:

- **Luồng xử lý (chạy nền):** 7 stage, chạy một lần cho mỗi Scene.
- **Luồng Query:** chạy theo từng câu hỏi và không đụng tới 6 stage đầu.

Tài liệu này được viết từ việc đọc code tại commit `f285cc9` và notebook, chưa đối chiếu với một lần chạy thật. Khi code và tài liệu khác nhau, hãy tin code.

## Cấu hình đang dùng

[colab_server.ipynb](../notebooks/colab_server.ipynb) chạy `pipeline: "real"` với `view_stride: 10`. Ngoài `storage_root` và giới hạn 1–2000 ảnh, các thiết lập còn lại lấy mặc định từ [config.py](../src/scenetiq/config.py).

Chạy local theo lệnh trong `AGENTS.md` thì dùng `FakePipeline`, không có model thật.

```text
Trình duyệt → Cloudflare Tunnel → FastAPI trên Colab
Ảnh → Reconstructing → Fusing → Aligning → Superpoints → Masking → Merging → Embedding → Captioning → lưu đĩa
Câu Query → parser trên Qwen3-VL → (không có Relation) SigLIP2 text + Qwen3-Embedding → hai bảng xếp hạng → RRF → top-K Match
                                 → (có Relation) xếp hạng như trên cho từng vật → giải Relation trên hình học đã lưu → Match kèm Anchor
```

## Upload và khởi động job

Phần này nằm trong [api.py](../src/scenetiq/api.py) và [jobs.py](../src/scenetiq/jobs.py):

1. **`POST /captures`:** khai báo số ảnh. Server từ chối ngay nếu số ảnh sai giới hạn hoặc đang có job chạy.
2. **`POST /captures/{id}/views`:** gửi từng ảnh một, vì Cloudflare chặn body lớn (ADR 0006). Chỉ nhận JPG và PNG.
3. **`POST /scenes`:** kiểm tra đã đủ ảnh rồi lấy `paths[::10]`. Ví dụ 100 ảnh thì chỉ dùng 10 View: số 0, 10, …, 90.
4. **Chạy job:** mỗi lần chỉ một job, chạy trên thread nền. Frontend theo dõi qua `GET /jobs/{id}`.

## 8 stage

Thứ tự nằm trong `RealPipeline.run` của [real.py](../src/scenetiq/pipeline/real.py).

| # | Stage | Công cụ | Đầu ra |
|---|---|---|---|
| 1 | Reconstructing | Depth Anything 3 | Độ sâu theo mét và camera của từng View |
| 2 | Fusing | TSDF của Open3D (CPU) | Mesh; các đỉnh chính là điểm của Reconstruction |
| 3 | Aligning | GeoCalib + RANSAC | Hình học và camera đã xoay, hướng lên là −Y |
| 4 | Superpoints | ScanNet Segmentator | Id superpoint của từng điểm |
| 5 | Masking | SAM2 | Các Mask không nhãn của từng View |
| 6 | Merging | MV3DIS | Id Instance của từng điểm |
| 7 | Embedding | SigLIP2 | Một vector ảnh của từng Instance |
| 8 | Captioning | Qwen3-VL-8B + Qwen3-Embedding | Caption (Name, Synonyms, 4 Description) và một vector cho mỗi Description của từng Instance (nếu đọc được) |

### 1. Reconstructing

Code: [reconstruct.py](../src/scenetiq/pipeline/reconstruct.py)

- DA3 xử lý **mọi View trong một lượt forward**, nên Capture quá lớn sẽ hết VRAM (OOM).
- Cạnh dài của ảnh được giảm về tối đa 504 px. Các stage đến hết Merging đều làm việc ở độ phân giải này.
- Độ sâu của một pixel bị đặt về 0 nếu nó thuộc một trong các trường hợp:
  - nằm dưới phân vị 30 về độ tin cậy, tính trên toàn Capture;
  - xa hơn 12 m;
  - không hợp lệ (không hữu hạn hoặc không dương).
- Xong stage, DA3 được giải phóng khỏi GPU, kể cả khi lỗi.

### 2. Fusing

Code: [fusion.py](../src/scenetiq/pipeline/fusion.py)

- Độ sâu của mọi View được tích hợp vào lưới TSDF với voxel 2 cm và ngưỡng cắt 4 voxel, rồi trích ra mesh.
- `weight_threshold=0` giữ mọi voxel đã quan sát, nên vùng chỉ một View nhìn thấy vẫn còn.
- Màu điểm lấy từ ảnh đã giảm độ phân giải.
- **Thứ tự điểm là hợp đồng:** PLY, `instance_ids.bin` và đầu vào MV3DIS đều dùng thứ tự đỉnh này. Sau bước này không được lọc hay sắp xếp lại điểm.

### 3. Aligning

Code: [alignment.py](../src/scenetiq/pipeline/alignment.py)

1. Chọn tối đa 12 View cách đều và chạy GeoCalib riêng từng ảnh, kèm focal từ DA3 làm gợi ý.
2. Đổi các vector hướng lên sang hệ thế giới rồi tìm đồng thuận: **hơn 50%** số ước lượng phải nằm trong góc 10°.
3. Nếu không có đồng thuận, giữ nguyên Scene và job vẫn chạy tiếp.
4. Nếu có, tìm tối đa 5 mặt phẳng bằng RANSAC. Mặt lớn nhất chứa ít nhất 5% số điểm đã tìm và lệch không quá 10° sẽ tinh chỉnh hướng lên. Sàn, mặt bàn hay trần đều được tính.
5. Xoay điểm và camera để hướng lên thành −Y. Độ sâu không đổi vì nó đo trong hệ camera.

### 4. Superpoints

Code: [superpoints.py](../src/scenetiq/pipeline/superpoints.py)

Nhóm các đỉnh mesh theo độ tương đồng pháp tuyến (`k_thresh=0.01`, tối thiểu 20 đỉnh). MV3DIS nhờ vậy làm việc trên các nhóm điểm thay vì từng điểm. Đây là stage cuối dùng tam giác mesh; kết quả lưu ra không có mesh.

### 5. Masking

Code: [masking.py](../src/scenetiq/pipeline/masking.py)

- SAM2 tự sinh Mask từ lưới điểm 32×32, không dùng văn bản, nên việc tách Instance không phụ thuộc từ vựng nào.
- Mask bị loại nếu `pred_iou < 0.8`, `stability < 0.9` hoặc diện tích dưới 0,1% ảnh.
- Mỗi View giữ tối đa 255 Mask điểm cao nhất, vì MV3DIS lưu nhãn pixel bằng `uint8`.

### 6. Merging

Code: [merging.py](../src/scenetiq/pipeline/merging.py), [mv3dis_base.py](../src/scenetiq/vendor/mv3dis/mv3dis_base.py)

Đây là bước chuyển Mask 2D thành Instance 3D:

1. **Chuẩn bị:**
   - Lọc Mask chồng lấn (IoU > 0.5) trong từng View và tạo bản đồ nhãn theo pixel.
   - Chiếu mọi điểm vào mọi View, tạo các ma trận N×M trên RAM. Điểm được tính là nhìn thấy khi nằm trong ảnh và lệch độ sâu không quá 5%.
2. **Gom vùng lần đầu:** chấm độ tương đồng giữa các superpoint lân cận dựa trên việc chúng có rơi vào cùng Mask ở các View hay không. Vùng được gom khi affinity đạt từ 0.5, có trọng số theo số điểm và nghịch đảo khoảng cách.
3. **Tinh chỉnh:** dùng các vùng vừa có để chấm lại Mask ở những View thấy trên 30% số điểm của vùng, lọc lại, tạo lại bản đồ nhãn rồi gom vùng thêm hai lượt.
4. **Dọn dẹp:** vùng nhỏ dưới 200 điểm được nhập vào hàng xóm giống nhất, không có hàng xóm phù hợp thì bỏ.
5. **Đổi nhãn:** vùng dưới 50 điểm bị bỏ. Các vùng còn lại được đánh id theo kích thước giảm dần, điểm không thuộc Instance nào có id −1.

`compute_mask_weights` và hàm tính ma trận tương đồng chưa được đọc hết khi viết tài liệu này, nên hai ý liên quan chỉ ở mức ý tưởng.

### 7. Embedding

Code: [embedding.py](../src/scenetiq/pipeline/embedding.py)

Với mỗi Instance:

1. Tìm các điểm thấy được ở từng View, có kiểm tra bị che bằng độ sâu.
2. Chọn tối đa 5 View thấy nhiều điểm nhất, mỗi View thấy ít nhất 30 điểm.
3. Cắt từ **ảnh gốc độ phân giải đầy đủ** với 2 mức mở rộng (10% và 40%), tức tối đa 10 crop. Crop là hình chữ nhật nên có cả nền xung quanh.
4. Mã hóa bằng SigLIP2, lấy trung bình rồi chuẩn hóa L2.

Instance không có View nào đạt ngưỡng sẽ bị bỏ, điểm của nó về −1. Vì vậy id Instance cuối cùng có thể bị ngắt quãng. SigLIP2 ở lại GPU để phục vụ Query.

Việc chọn View (bước 1–2) do `plan_views` làm một lần, rồi dùng chung cho stage Captioning.

### 8. Captioning

Code: [captioning.py](../src/scenetiq/pipeline/captioning.py), [qwen_vl.py](../src/scenetiq/pipeline/qwen_vl.py), [qwen_embedding.py](../src/scenetiq/pipeline/qwen_embedding.py). Quyết định: ADR 0008.

Với mỗi Instance còn lại sau Embedding:

1. Lấy tối đa 3 View tốt nhất trong kế hoạch của Embedding.
2. Cắt từ ảnh gốc với 60% bối cảnh quanh Instance, vẽ khung đỏ quanh Instance, rồi thu nhỏ về tối đa 448 × 448 px.
3. Gửi các crop trong **một prompt** cho Qwen3-VL-8B, yêu cầu trả JSON bằng tiếng Anh gồm `name`, `synonyms` và **một câu (Description) cho mỗi Aspect**: `identity` (là gì, tên gọi khác), `appearance` (màu, hình dáng, kích thước), `material` (chất liệu, bộ phận), `use` (công dụng). Description được để trống nếu ảnh không cho thấy Aspect đó. Câu trả lời bị **ép theo một JSON schema cố định** bằng xgrammar ngay khi sinh từng token (đủ mọi key, tối đa 3 synonym, mỗi chuỗi có giới hạn độ dài), nên chỉ còn một cách mất Caption: bị cắt ở `max_new_tokens`.
4. Đọc JSON một cách dễ dãi (chấp nhận code fence, chữ thừa), giữ tối đa 3 synonym. Câu trả lời không có `name` thì Instance không có Caption, lý do được ghi log, job vẫn chạy tiếp. `identity` trống thì được viết từ Name và Synonyms, ví dụ `"A sofa, also called a couch or settee."`.
5. Mã hóa **từng Description không trống thành một vector riêng** bằng Qwen3-Embedding.

Qwen3-VL ở lại GPU sau stage, kể cả khi lỗi, cùng Qwen3-Embedding và SigLIP2 để phục vụ Query. Model được nạp ở lần dùng đầu tiên (bởi job hoặc Query) và mỗi lần sinh giữ khóa của nó, nên một Query chỉ phải chờ tối đa một Instance. Vì vậy job bắt đầu với ít hơn khoảng 17 GB bộ nhớ GPU trống cho DA3 và SAM2. Mỗi job ghi một dòng `captioning captioned=… uncaptioned=…` lên notebook.

## Lưu kết quả

Sau stage 8, centroid và bounding box được tính, rồi [artifacts.py](../src/scenetiq/artifacts.py) ghi 5 file:

| File | Nội dung |
|---|---|
| `points.ply` | Tọa độ xyz theo mét và màu rgb |
| `instance_ids.bin` | Id Instance của từng điểm, cùng thứ tự với PLY |
| `instances.json` | Id, centroid, bounding box và Caption (Name, Synonyms, 4 Description) hoặc null |
| `embeddings.npy` | Vector ảnh của từng Instance |
| `caption_embeddings.npy` | Shape (Instance, Aspect, D): một vector cho mỗi Description theo thứ tự Aspect; hàng 0 cho Description trống hoặc Instance không có Caption. Không có file nếu không Instance nào có Caption |

Dữ liệu nằm ở `/content/scenetiq-data`, nên **mất khi runtime Colab kết thúc**.

## Luồng Query

Luồng này nằm trong [query.py](../src/scenetiq/query.py). Với câu `"red chair"`:

1. Query ngắn (≤ `siglip.prompt_max_words` = 4 từ) được bọc thành `"this is a photo of red chair."`; Query dài hơn là một câu nên giữ nguyên văn. Cả hai được chuyển thành chữ thường, cắt ở 64 token (giới hạn phía text của SigLIP2, có ghi log khi cắt) rồi mã hóa bằng SigLIP2. Nếu Scene có Caption, câu gốc cũng được mã hóa bằng Qwen3-Embedding, có thêm query instruction; Scene không có Caption thì không cần nạp model này.
2. Xếp hạng mọi Instance theo cosine với vector ảnh, và xếp hạng các Instance có Caption theo cosine của **Description giống Query nhất** (max trên các Description của nó).
3. Gộp hai bảng bằng Reciprocal Rank Fusion: mỗi bảng cộng `1 / (rrf_k + hạng)`, với `rrf_k = 60`. Instance không có Caption chỉ có phần của bảng ảnh.
4. Trả `top_k` Instance đứng đầu: số Query yêu cầu, tối đa `query_max_top_k` (mặc định 20), hoặc `query_top_k` (mặc định 5) khi Query không yêu cầu. Viewer tô các Match bằng heatmap kéo giãn theo điểm của các Match đang hiện: Match tốt nhất màu nóng nhất, Match thấp nhất màu lạnh nhất. Điểm được chia cho giá trị lớn nhất có thể, `số bảng / (rrf_k + 1)`, nên nằm trong [0, 1] nhưng chỉ có nghĩa trong một Query. Mỗi Match kèm `image_score`, `caption_score` (cosine thô của Description tốt nhất) và `caption_aspect` (Aspect của Description đó) để debug.
5. Không có ngưỡng: Scene có Instance thì luôn có Match. Scene không có Instance thì trả danh sách rỗng. Scene không Instance nào có Caption thì chỉ xếp theo ảnh.

Mọi Query trước hết đi qua parser và có thể được giải theo Relation ([ADR 0009](adr/0009-spatial-queries-parsed-and-solved.md)). Parser nằm trong [query_parser.py](../src/scenetiq/query_parser.py) và chạy trên chính Qwen3-VL của stage Captioning (cùng một instance với Pipeline):

1. Gửi một prompt **chỉ có chữ**, bằng tiếng Anh, kèm **nguyên văn Query, không cắt** (giới hạn 64 token chỉ áp dụng cho SigLIP2), đặt ở dòng cuối dưới dạng một chuỗi JSON để dấu nháy hay câu lệnh trong Query không phá được prompt. Prompt liệt kê 13 Relation, mỗi cái một dòng nghĩa, yêu cầu mỗi vật là một cụm từ tiếng Anh ngắn (chỉ tả vật đó, không kèm Relation hay vật khác) dù Query viết bằng tiếng Việt, và đưa những gì tập Relation không diễn đạt được (trái, phải, trước, sau, phủ định, đếm, thứ tự) vào `unsupported`, trích nguyên văn. Không có ví dụ mẫu, vì model sẽ chép lại (ADR 0008).
2. Câu trả lời bị ép theo một JSON schema cố định (`objects`, `target`, `constraints`, `unsupported`) bằng xgrammar, với một logits processor mới cho mỗi lần sinh. Giới hạn là `query_parser.max_new_tokens` = 1024, riêng với Captioning.
3. Kiểm tra lần hai: câu trả lời bị cắt hoặc không đọc được, id vật lặp lại, `target` hay vật trong constraint không có trong danh sách, Relation sai số Anchor, hoặc vật làm Anchor của chính nó đều là parse thất bại.

Vì vậy **mọi Query phải chờ một lần sinh**, kể cả Query không có Relation, và Query đầu tiên sau khi khởi động lại còn phải chờ nạp Qwen3-VL. Trong lúc job đang Captioning, Query chen vào giữa hai Instance. Chạy local thì dùng parser giả, không có model.

Response có thêm `relations_applied` (Relation đã quyết định Match), `relations_stated` (Query sau khi parse có Relation), `unsupported` (những phần Query mà tập Relation không diễn đạt được, trích nguyên văn theo parser) và `parse_failed` (parser không đọc được Query). Phần không hỗ trợ chỉ được báo lại, các Relation còn lại vẫn được áp dụng. Khi parse thất bại, hoặc có Relation nhưng không có phép gán hợp lệ (ví dụ Scene có ít Instance hơn số vật trong Query, lúc đó `relations_stated` đúng còn `relations_applied` sai), Query chạy như các bước trên với câu gốc và vẫn trả 200. Lý do parse thất bại được ghi log lên notebook.

Frontend tải `points`, `instance-ids` và `instances` (kèm `name` của Caption) qua [api.ts](../frontend/src/api.ts). [SceneViewer.tsx](../frontend/src/components/SceneViewer.tsx) hiện Name cạnh điểm trong danh sách Match và làm nhãn trên bounding box, đánh dấu các Anchor bằng màu riêng, và hiện một dòng thông báo khi có phần Query không được tính đến hoặc parse thất bại.

## Khi nào job thất bại

Job chuyển sang `failed` và lưu thông báo lỗi khi một stage ném ngoại lệ, ví dụ:

- hết bộ nhớ GPU;
- không View nào còn độ sâu tin cậy, hoặc mesh rỗng;
- Scene có dưới 8 điểm hoặc dưới 2 primitive;
- không View nào có Mask.

Nếu server khởi động lại giữa chừng, job dở dang cũng bị đánh dấu `failed`.

## Những điểm yếu cần biết

- **Match luôn có, kể cả khi không vật nào khớp:** top-K không có ngưỡng, và điểm RRF chỉ so được trong một Query. `query_top_k`, `rrf_k` và các thiết lập Captioning chưa được kiểm chứng; repo chưa có công cụ đo trên ground truth.
- **Instance không có Caption bị xếp thấp hơn:** nó chỉ có một phần trong RRF.
- **EXIF không được áp dụng:** ảnh dọc lưu dạng ảnh ngang kèm cờ EXIF sẽ đi qua mọi stage trong tư thế nằm ngang.
- **Ảnh xoay nhiều có thể làm sai Gravity alignment:** GeoCalib chỉ đúng đến khoảng ±45°. Nếu mọi View xoay giống nhau, bước đồng thuận không phát hiện được.
- **Không có kiểm tra trước về VRAM:** chỉ số ảnh upload bị giới hạn, còn số View vượt sức GPU thì job chỉ lỗi OOM khi chạy.
- **Parser chưa được chạy thử trên Colab:** thời gian parse mỗi Query, thời gian nạp ở Query đầu tiên, việc đọc Query tiếng Việt, và việc sinh chỉ có chữ với cùng kích thước vocabulary của xgrammar đều chưa được đo. Câu trả lời parse sai mà vẫn hợp lệ (ví dụ chọn nhầm Relation) không bị phát hiện, chỉ thấy qua Anchor trên viewer.
- **`view_stride: 10` bỏ 90% số ảnh:** các ảnh này vẫn được lưu nhưng không dùng ở stage nào, kể cả Embedding.
