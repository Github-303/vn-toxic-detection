# Vietnamese Toxic Comment Detection API

Hệ thống API phát hiện và phân loại bình luận độc hại trong tiếng Việt sử dụng Deep Learning.

## 📋 Tính năng

- ✨ Phát hiện và phân loại bình luận độc hại tự động
- 🚀 API bất đồng bộ hiệu năng cao với FastAPI
- 🔒 Xác thực và phân quyền người dùng
- 📊 Theo dõi và thống kê kết quả phân tích
- 🔄 Tích hợp với nhiều nền tảng mạng xã hội
- 📈 Monitoring và metrics tích hợp sẵn

## 🏗 Kiến trúc

Dự án được xây dựng theo mô hình Model-Controller-Presenter (MCP):

```
app/
├── config/         # Cấu hình ứng dụng
├── controllers/    # Business logic
├── middleware/     # Middleware components
├── models/         # Database models
├── ml/            # Machine learning components
├── presenters/    # API endpoints
├── schemas/       # Pydantic schemas
└── utils/         # Utility functions
```

## 🛠 Công nghệ sử dụng

- **FastAPI**: Web framework
- **SQLAlchemy**: ORM và database toolkit
- **PostgreSQL + pgvector**: Database với hỗ trợ vector operations
- **Redis**: Caching và rate limiting
- **PyTorch**: Deep learning framework
- **Docker**: Containerization

## 🚀 Hướng dẫn cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- PostgreSQL 13+
- Redis 6+
- Docker và Docker Compose (tùy chọn)

### Cài đặt thông thường

1. Clone repository:
```bash
git clone https://github.com/your-username/vihsd.git
cd vihsd
```

2. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

4. Thiết lập môi trường:
```bash
cp .env.example .env
# Chỉnh sửa các biến trong .env theo môi trường của bạn
```

5. Khởi tạo database:
```bash
# Đảm bảo PostgreSQL đang chạy
psql -U postgres
CREATE DATABASE comment_db;
CREATE EXTENSION IF NOT EXISTS vector;
```

6. Chạy ứng dụng:
```bash
uvicorn app.main:app --reload
```

### Cài đặt với Docker

1. Chuẩn bị môi trường:
```bash
cp .env.example .env
cp docker-compose.example.yml docker-compose.yml
# Chỉnh sửa các biến trong .env
```

2. Build và chạy containers:
```bash
docker-compose up -d
```

## 📚 Sử dụng API

### Authentication

1. Đăng ký tài khoản:
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
     -H "Content-Type: application/json" \
     -d '{"username": "user", "email": "user@example.com", "password": "password123"}'
```

2. Đăng nhập:
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "password123"}'
```

### Phân tích bình luận

```bash
curl -X POST "http://localhost:8000/api/v1/detection/analyze" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"text": "Nội dung bình luận", "platform": "web"}'
```

Performance metrics for each model:

| Model     | Accuracy | F1-Score | Processing Time |
|-----------|----------|----------|----------------|
| PhoBERT   | 0.92     | 0.91     | 120ms         |
| BERT4News | 0.90     | 0.89     | 115ms         |
| TextCNN   | 0.87     | 0.86     | 45ms          |
| LSTM      | 0.86     | 0.85     | 55ms          |
| GRU       | 0.85     | 0.84     | 50ms          |

- `DATABASE_URL`: URL kết nối PostgreSQL
- `REDIS_URL`: URL kết nối Redis
- `JWT_SECRET_KEY`: Khóa bí mật cho JWT
- `MODEL_NAME`: Tên model sử dụng (phobert, bert4news, etc.)
- Xem thêm trong file `.env.example`

### Model Configuration

Hỗ trợ nhiều loại model:
- PhoBERT: Transformer cho tiếng Việt
- BERT4News: BERT tiếng Việt từ báo chí
- TextCNN: Mô hình CNN cho văn bản
- BiLSTM: Mô hình LSTM hai chiều

## 📊 Monitoring

- Health check: `GET /api/v1/health`
- Metrics: `GET /api/v1/metrics`
- Prometheus metrics: `GET /metrics`

## 🧪 Testing

Chạy unit tests:
```bash
pytest
```

Chạy với coverage:
```bash
pytest --cov=app tests/
```

## 📝 API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/api/v1/openapi.json`

## 🤝 Đóng góp

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/amazing-feature`)
3. Commit thay đổi (`git commit -m 'Add amazing feature'`)
4. Push lên branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

## 📄 License

MIT License - xem [LICENSE](LICENSE) để biết thêm chi tiết.