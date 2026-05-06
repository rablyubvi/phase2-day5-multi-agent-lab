# Design Template

## Vấn đề

Xây dựng một trợ lý nghiên cứu có khả năng nhận một truy vấn kỹ thuật, thu thập bằng chứng hỗ trợ, đánh giá độ tin cậy của bằng chứng, và tạo ra câu trả lời cuối cùng kèm theo trích dẫn rõ ràng, có thể truy vết.

---

## Tại sao cần multi-agent?

Mô hình single-agent vẫn có thể trả lời câu hỏi, nhưng thường trộn lẫn các bước: truy xuất dữ liệu, đánh giá và tổng hợp trong một lần xử lý.

Việc tách thành nhiều agent giúp:

* Phân tách rõ vai trò
* Dễ debug khi có lỗi
* Có checkpoint riêng cho:

  * chất lượng nguồn
  * chất lượng phân tích
  * chất lượng câu trả lời cuối

---

## Vai trò các Agent

| Agent      | Nhiệm vụ                                   | Input                     | Output                      | Failure mode                    |
| ---------- | ------------------------------------------ | ------------------------- | --------------------------- | ------------------------------- |
| Supervisor | Điều phối workflow và quyết định dừng      | `ResearchState`           | Lịch sử route và trace      | Loop quá lâu hoặc dừng quá sớm  |
| Researcher | Thu thập nguồn và ghi chú nghiên cứu       | Query + giới hạn nguồn    | `sources`, `research_notes` | Nguồn yếu hoặc hallucination    |
| Analyst    | Đánh giá chất lượng và phân loại thông tin | Research notes + sources  | `analysis_notes`            | Đánh giá sai độ tin cậy         |
| Writer     | Tổng hợp câu trả lời cuối                  | Research + analysis notes | `final_answer`              | Trôi chảy nhưng thiếu dẫn chứng |
| Critic     | Kiểm tra lỗi và độ phủ citation            | Final answer + sources    | `critic_notes`, `errors`    | Bỏ sót lỗi tinh vi              |

---

## Shared State

`ResearchState` lưu toàn bộ trạng thái xuyên suốt pipeline:

* `request`: query gốc, số lượng nguồn, audience
* `iteration`, `route_history`: điều khiển luồng
* `sources`: dữ liệu từ researcher
* `research_notes`, `analysis_notes`, `final_answer`, `critic_notes`: output từng bước
* `agent_results`, `agent_outputs`: log structured
* `trace`: log từng bước execution
* `errors`: lỗi được phát hiện (phục vụ đánh giá)

---

## Routing Policy

```text
START -> Supervisor -> Researcher -> Supervisor -> Analyst -> Supervisor -> Writer -> Supervisor -> END
```

### Luật điều phối:

* Nếu chưa có research → chạy `Researcher`
* Nếu có research nhưng chưa có analysis → chạy `Analyst`
* Nếu có cả research và analysis → chạy `Writer`
* Nếu đã có `final_answer` hoặc đạt `max_iterations` → dừng

---

## Guardrails

* Max iterations: 6
* Timeout: 60 giây
* Retry: xử lý ở tầng LLM client
* Fallback:

  * dùng dữ liệu giả lập nếu không truy xuất được
  * heuristic synthesis nếu thiếu nguồn
* Validation:

  * critic review
  * kiểm tra citation trong trace

---

## Kế hoạch Benchmark

So sánh:

* Single-agent baseline
* Multi-agent workflow

---

### Metrics

* `latency_seconds`: thời gian xử lý
* `estimated_cost_usd`: chi phí ước tính
* `quality_score`: chất lượng câu trả lời
* `citation_coverage`: độ phủ trích dẫn
* `error_rate`: tỉ lệ lỗi

---

## Kỳ vọng kết quả

* Multi-agent:

  * Citation coverage cao hơn
  * Error rate thấp hơn
* Single-agent:

  * Nhanh hơn
  * Rẻ hơn với câu hỏi đơn giản
* Chất lượng:

  * Multi-agent vượt trội ở các câu hỏi phức tạp, nhiều bước suy luận

---
