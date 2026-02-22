FROM python:3.12-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境并安装依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制依赖文件并安装
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# 第二阶段：最终镜像
FROM python:3.12-slim

LABEL maintainer="coderxiu<coderxiu@qq.com>"
LABEL description="闲鱼AI客服机器人"
LABEL version="3.0"

# 设置时区和编码
ENV TZ=Asia/Shanghai \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HEADLESS=True

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo Asia/Shanghai > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 创建必要的目录
RUN mkdir -p data prompts

# 复制提示词文件
COPY prompts/tech_prompt.txt prompts/tech_prompt.txt
COPY prompts/default_prompt.txt prompts/default_prompt.txt

# 复制应用代码
COPY main.py XianyuAgent.py XianyuApis.py context_manager.py rag_service.py build_index.py ./
COPY utils/ utils/

CMD ["python", "main.py"]
