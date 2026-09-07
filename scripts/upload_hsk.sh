#!/bin/zsh
# 将 quant-lab 报告原址更新到 HSK 文件托管 (https://kpqv8z.gicp.fun)
# 密钥优先级: 环境变量 HSK_API_KEY (CI/GitHub Actions) > ~/.hsk/api_key.json (本地已配置)
# 资源 ID: 环境变量 HSK_RESOURCE_ID, 缺省 1788744899319110368

HSK="${HSK_CLI:-/Users/andy/.workbuddy/binaries/node/workspace/node_modules/.bin/hsk-cli}"
REPORT="$(cd "$(dirname "$0")/.." && pwd)/report/index.html"
RESOURCE_ID="${HSK_RESOURCE_ID:-1788744899319110368}"
LOG="$(cd "$(dirname "$0")/.." && pwd)/data/meta/upload_hsk.log"

# 本地缺省 hsk-cli 路径不存在时 (如 CI), 回退到 npx
if [ ! -x "$HSK" ]; then
    HSK="npx -y @aweray/hsk-cli@0.4.6"
fi

[ -f "$REPORT" ] || { echo "$(date '+%F %T') 报告不存在: $REPORT" >> "$LOG"; exit 1; }

KEY_ARGS=()
[ -n "$HSK_API_KEY" ] && KEY_ARGS=(--api-key "$HSK_API_KEY")

OUT=$($HSK host "$REPORT" --resource-id "$RESOURCE_ID" "${KEY_ARGS[@]}" --format json 2>/dev/null)

# hsk-cli 0.4.6 更新成功输出: update_result: {"code":0,"data":{},"message":"ok"}
if echo "$OUT" | grep -q '"message":"ok"'; then
    SIZE=$(stat -f %z "$REPORT" 2>/dev/null || stat -c %s "$REPORT")
    echo "$(date '+%F %T') 上传成功 -> https://kpqv8z.gicp.fun (${SIZE} bytes)" >> "$LOG"
else
    echo "$(date '+%F %T') 上传失败: $OUT" >> "$LOG"
    exit 1
fi
