# HUST Demand Forecasting Baseline
## Hướng dẫn reproduce

### Yêu cầu
- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2)** — Python package manager
    ```bash
    pip install uv
    # or
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
  - For **[Mac/Linux](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1)**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # or
    wget -qO- https://astral.sh/uv/install.sh | sh
    # or
    pip install uv
    ```
- **xelatex** — để compile báo cáo LaTeX (nếu cần)

### 1. Cài đặt môi trường

```bash
# Cài đặt dependencies
uv sync

# Activate virtual environment (Choose "datathon" kernel)
source .venv/bin/activate # Cho Mac/Linux
.venv\Scripts\activate       # Cho Windows 
```

