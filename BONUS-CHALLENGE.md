# Bonus Challenge — Build something real (UNGRADED)

> English: [`BONUS-CHALLENGE-EN.md`](BONUS-CHALLENGE-EN.md)

**Loại:** Sân chơi để **đem domain knowledge của bạn áp dụng vào 1 model align thật** — không có điểm số, không có deadline, không có rubric.
**Đối tượng:** Bạn vào vai *AI engineer* xây 1 aligned model cho 1 *audience cụ thể*, ship như sản phẩm.
**Effort target:** 4–8 giờ. Khuyến khích pair (2–3 người), brainstorm trước, code sau.
**Vibe coding khuyến khích:** code boilerplate AI lo, *domain choices + application objective* bạn tự nghĩ + viết.

> Đây là chỗ bạn ngừng coi DPO như bài toán giấy mà coi nó như **công cụ ship một thứ ai đó dùng được**. Phần thưởng thực sự: 1 portfolio piece có thể chỉ vào nói "tôi build cái này, audience là X, dùng để Y" — không phải "tôi tinh chỉnh β=0.05".

---

## 5 provocations — chọn 1, hoặc invent your own

Mỗi provocation có 4 phần: **Audience** (ai dùng) · **Domain knowledge** (bạn đem gì vào) · **Application objective** (model làm gì) · **Real-world output** (deliverable ship được).

### 1. Subject tutor cho môn bạn đang học

> *"DPO trên scaffolding pedagogy, không phải trên đáp án."*

- **Audience:** Học sinh THPT / sinh viên năm 1 đang ôn 1 môn cụ thể bạn giỏi (toán giải tích, hoá hữu cơ, lịch sử Việt Nam, lập trình Python, vật lý điện từ).
- **Domain knowledge:** Bạn biết 1 môn đủ sâu để phân biệt "trả lời sư phạm tốt" với "trả lời thẳng đáp án." Đó là yêu cầu cốt lõi.
- **Application objective:** Một model tutor không *cho* đáp án — gợi ý, đặt câu hỏi ngược, tham chiếu công thức từ SGK Việt Nam, tránh giải thích bằng English term học sinh chưa biết.
- **Real-world output:**
  - 200 cặp preference (prompt = câu hỏi học sinh điển hình; chosen = scaffolded VN tutoring response; rejected = direct answer in English mixed VN)
  - DPO adapter + GGUF Q4_K_M chạy trên CPU laptop học sinh (`llama-cpp-python`)
  - Mini Gradio demo (~50 dòng) để học sinh thử thật
  - Model card: "Math tutor cho học sinh lớp 12 chuẩn bị thi THPT — không thay sách giáo khoa"

**Câu hỏi để brainstorm:**
- Pedagogy tốt cho VN cấp 3 khác US Khan Academy thế nào? (cấu trúc đề thi, từ vựng SGK chuẩn)
- Khi nào *nên* trả thẳng đáp án (đã giải xong, học sinh muốn check)? Làm sao DPO learn được context đó?
- Cách nào đo "tutoring quality" mà không cần giáo viên thật chấm?

---

### 2. Customer-service chatbot cho 1 doanh nghiệp Việt thật

> *"Pick 1 cửa hàng. Tưởng tượng 200 câu khách hỏi. Build chatbot deploy được."*

- **Audience:** Khách hàng / chủ shop của 1 business cụ thể — quán cà phê Hà Nội, shop quần áo online, tiệm sửa xe máy, trường tiếng Anh nhỏ, homestay Sapa. Bạn pick.
- **Domain knowledge:** Bạn biết business đó hoạt động ra sao — giờ mở cửa, dịch vụ, giá tham khảo, cách nói chuyện đặc trưng (thân thiện vs lễ phép vs trẻ trung).
- **Application objective:** Chatbot trả lời FAQ on-brand, đưa next step rõ (số điện thoại, link đặt, địa chỉ Google Maps), không tự bịa thông tin không có.
- **Real-world output:**
  - 200 cặp preference (chosen = on-brand voice + có contact CTA; rejected = generic AI khô khan, hoặc trả về English)
  - DPO adapter + GGUF
  - Deployable: 1 file `serve.py` 30 dòng dùng `llama-cpp-python` + FastAPI endpoint `/chat`
  - Model card có "What I built this for: <business>" + 5 sample conversations

**Câu hỏi để brainstorm:**
- Bạn xin được 200 câu khách thật không? (chủ shop có FB inbox bạn xem được? hỏi bạn của bạn ai có shop?)
- Tone: "anh/chị" formal vs "bạn" casual vs xưng "shop/em" theo style cafe trẻ — DPO learn được không?
- Khi khách hỏi ngoài scope (giá đối thủ, review tiêu cực), chatbot nên làm gì? Refuse? Defuse? Redirect chủ shop?

---

### 3. Job-shadow assistant — model cho 1 nghề bạn quan sát hằng ngày

> *"Cho người không-phải-bạn dùng. Bạn build cho tài xế Grab, không phải cho sinh viên VinUni."*

- **Audience:** Một vai trò lao động Việt Nam cụ thể — Grab/be driver, shipper food, lễ tân khách sạn, tiểu thương chợ đầu mối, content creator TikTok, thợ điện nước khu phố.
- **Domain knowledge:** Bạn quan sát/phỏng vấn ai đó làm nghề này. Nghề đó có ngôn ngữ, áp lực thời gian, decision pattern khác văn phòng.
- **Application objective:** Model trả lời câu hỏi điển hình của *người làm nghề đó*, không phải câu hỏi sinh viên về nghề đó. Nhanh, action-oriented, không lecture, hiểu ngữ cảnh nghề.
- **Real-world output:**
  - 200 cặp preference (prompt = câu hỏi nghề thật bạn quan sát; chosen = ngắn, cho được hành động kế tiếp ngay, hiểu bối cảnh; rejected = câu trả lời 5 đoạn lecture-style của ChatGPT mặc định)
  - DPO adapter + GGUF (mục đích: chạy trên điện thoại Android tầm trung, vì người dùng có thể không có laptop)
  - Mini demo: command-line script bạn ngồi cùng người làm nghề đó test 5–10 prompt thật
  - Model card: "Cho ai" + "Tôi đã ngồi quan sát X giờ với người làm nghề này, đây là 5 quyết định design dựa trên đó"

**Câu hỏi để brainstorm:**
- Dialogue length: tài xế đang lái không đọc được 200 từ. Làm sao DPO ép response < 50 từ?
- Vốn từ ngành (jargon shipper: "boom đơn", "khách bom") — model có hiểu không? Cần data thế nào?
- Bạn có thể *test với 1 người làm nghề thật* không? Nếu có — đó là evaluation tốt nhất, hơn mọi GPT-4 judge.

---

### 4. Domain-safe assistant — model có boundary rõ trong 1 lĩnh vực nhạy cảm

> *"Quan trọng nhất: model không từ chối quá nhiều, không lơ ngơ quá nhiều. Bạn build alignment cho 1 ngành cụ thể."*

- **Audience:** Người dùng phổ thông Việt Nam tìm thông tin 1 lĩnh vực có rủi ro — sức khoẻ tinh thần (mental health support), thông tin pháp lý cơ bản (luật lao động, dân sự), thông tin thuốc cơ bản, kế hoạch tài chính cá nhân, info-bot cho tiêm chủng/HIV-prevention.
- **Domain knowledge:** Bạn biết (hoặc research) gì là *thông tin bổ ích* vs *advice phải để cho chuyên gia*. Bạn biết hotlines / nguồn VN chính thức (1900-XXXX cho chuyên gia, Bộ Y tế, Tổng đài Phụ nữ).
- **Application objective:** Model:
  1. Cung cấp thông tin nền + nguồn chính thức
  2. Không tự đưa ra advice cá nhân hoá (kê thuốc, chẩn đoán, tư vấn pháp lý)
  3. Empathetic — không lạnh lùng đẩy ra
  4. Luôn refer tới chuyên gia + cho hotline VN khi cần
- **Real-world output:**
  - 200 cặp preference (chosen = info + nguồn + hotline + soft handoff; rejected = direct advice OR cold refusal)
  - DPO adapter + GGUF
  - **Model card đặc biệt:** explicit "What this model will NOT do" list — đây là phần quan trọng nhất, document boundary rõ
  - Test set 20 prompt: 10 benign-but-sensitive (model NÊN trả lời với resource), 10 boundary-crossing (model PHẢI refuse + handoff). Report precision/recall trên 2 tập đó.

**Câu hỏi để brainstorm:**
- "Empathetic refuse" là gì trong tiếng Việt? Khác US "I can't help with that" thế nào?
- Đối tượng có thể đang stress nặng — DPO có thể vô tình train model thành "robotic" không? Làm sao tránh?
- Bạn có thể tham khảo Hội Tâm lý học VN, Tổng đài 1900-1567, hoặc model card của Anthropic Claude cho constitutional AI làm reference.

---

### 5. Style-mimic — model viết kiểu một người bạn admire

> *"Pick 1 người. 50 samples. DPO style transfer."*

- **Audience:** Bản thân bạn (writing assistant), hoặc 1 brand/người bạn fan (ghostwriter style mimic), hoặc tổ chức (báo Tuổi Trẻ, VTV24, kênh TikTok cụ thể).
- **Domain knowledge:** Bạn đọc đủ nhiều văn của người/brand đó để *cảm* được style — câu dài hay ngắn, chữ Hán Việt nhiều hay ít, thái độ humour vs serious, có dùng emoji không, paragraph 1 câu hay 5 câu.
- **Application objective:** Model viết tiếng Việt theo style đó — không phải copy nội dung, mà *match register*.
- **Real-world output:**
  - 50–100 samples văn của target style (essay, blog, tweet thread, voice memo transcript)
  - 200 cặp preference (prompt = "viết 1 đoạn về <topic> theo style <X>"; chosen = style match; rejected = generic AI flat tone). *Chosen có thể chính là rewrites bạn tự edit — đó là deepest domain knowledge bạn đem vào.*
  - DPO adapter
  - 5 side-by-side outputs: prompt cùng nội dung, output trước/sau DPO, đem cho 1 fan của style đó *blind-test* xem có nhận ra không
  - (Tuỳ chọn) HF Hub push với model card ghi rõ inspiration source + license/attribution

**Câu hỏi để brainstorm:**
- Style transfer khác content imitation thế nào? DPO có thể tách không, hay sẽ học cả nội dung lẫn style?
- Đạo đức: nếu bạn mimic 1 nhà văn còn sống, đó là homage hay vi phạm? Model card của bạn nói gì về điều đó?
- 50 samples đủ không? Có nghiên cứu cho thấy style transfer cần ít data hơn content learning — bạn empirically check.

---

## Hoặc — invent your own

Cấu trúc tốt cho 1 provocation tự tạo, đủ 4 trục:

```
AUDIENCE:           ai sẽ dùng model? (cụ thể, không phải "everyone")
DOMAIN KNOWLEDGE:   bạn đem gì vào? (kinh nghiệm cá nhân, kỹ năng,
                    quan hệ với người trong ngành, dữ liệu bạn có
                    quyền access)
APPLICATION OBJECT: model làm GÌ cho audience? (1 use case rõ, không
                    phải "improve helpfulness" chung chung)
REAL-WORLD OUTPUT:  shape của deliverable (GGUF? Gradio? CLI? FastAPI?
                    HF Hub model card? Tích hợp vào app/web có sẵn?)
```

Một số nhánh khác đáng explore:

- **Accessibility-first:** model output thân thiện screen reader, câu ngắn cho người ít chữ, hỗ trợ ESL learner học tiếng Việt
- **Cultural register:** train ưu tiên Bắc vs Nam, formal vs casual, register theo ngữ cảnh (chat sếp vs chat bạn)
- **Niche cultural archive:** thơ Việt, lịch sử triều đại, ẩm thực vùng miền, văn học dân gian — model không hallucinate VN history
- **Code-review assistant cho 1 framework:** FastAPI/React/Laravel idiomatic comments
- **Voice-to-action:** transcript → action, ví dụ ghi chú họp → list todo

Tự build provocation thường ra portfolio piece có chiều sâu hơn 5 provocation gợi ý — vì *bạn* care, không phải *tôi* care.

---

## Self-checklist cho strong submission

Strong submissions thường có:

- [ ] **`bonus/README.md` ≥ 400 từ** trả 4 trục: Audience / Domain knowledge / Application objective / Real-world output. Trang đầu đọc xong thấy "ai sẽ dùng cái này, làm gì."
- [ ] **Preference data từ domain bạn**, không phải UltraFeedback translate. Ít nhất 100/200 cặp do bạn tự construct với judgment domain.
- [ ] **Deliverable chạy được**: ai đó (TA, bạn cùng lớp, người ngoài) clone repo + chạy 1 lệnh + tương tác với model.
- [ ] **Model card** với explicit "What this model is for / NOT for / Known limitations" — đặc biệt quan trọng cho provocation 4.
- [ ] **5 sample interactions** trong README cho người đọc thấy quality, không phải số liệu.
- [ ] **Honest limitations**: 1 đoạn "What this POC doesn't handle yet" — privacy, bias, scale, license.
- [ ] **(Tuỳ chọn) HF Hub push** với README đầy đủ — quy ước: tên model có chữ "v0" hoặc "experimental" để không gây hiểu nhầm production-ready.

---

## Format chấp nhận

Free-form. Suggested structure:

```
bonus/
├── README.md            # 1 trang: Audience, Domain, Objective, Output
├── data/
│   ├── prompts.jsonl    # 200 prompts của domain bạn
│   └── pairs.parquet    # preference pairs
├── train.py (or .ipynb) # training run (re-use scripts/train_dpo.py với data của bạn)
├── adapters/dpo-bonus/  # gitignore — output adapter
├── demo/
│   ├── serve.py         # 1 file deployable (FastAPI / Gradio / CLI)
│   └── 5-samples.md     # 5 prompt + output trước/sau DPO
└── MODEL-CARD.md        # ship-ready documentation
```

**Pair / triple work khuyến khích** — chú thích contributors trong `bonus/README.md` đầu file.

**Vibe coding workflow log** — nếu bạn dùng AI heavily, ghi ngắn (~100 từ) trong `MODEL-CARD.md` "1 prompt nào hiệu quả nhất, 1 prompt nào fail."

---

## Submission

Add folder `bonus/` vào repo public của bạn (cùng repo Lab 22 chính). Trong `submission/REFLECTION.md`, mention bạn đã làm bonus. Grader review từ cùng URL public LMS.

> Bonus **không** ảnh hưởng core grade. Một strong bonus submission được instructor đọc kỹ + comment dài về *judgment* + *application thinking* của bạn — không về β/loss/reward gap.

---

## "Ai cũng có 1 domain"

Bạn không cần làm AI researcher full-time để build cái có ý nghĩa. Bạn biết 1 môn học, biết 1 nghề, biết 1 cộng đồng, biết style của 1 người. Đó đã là domain knowledge đủ để aligned model của bạn *phục vụ ai đó cụ thể* hơn là một generic ChatBot copy thứ 1001.

Mở PR vào repo gốc nếu muốn share với cohort sau.
