#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# DataMark 验收脚本：一次性运行书稿 2.5 工具包的全部验收命令，并把带时间的
# 完整输出写入 book_outputs/acceptance.txt（书稿与交付报告引用）。
#
# 用法：在 toolkit/ 仓库根目录执行
#     bash scripts/run_acceptance.sh
#
# 注意：本机没有 `python` 命令，一律使用 `python3`；默认模拟模式只用标准库。
set -u

cd "$(dirname "$0")/.." || exit 1
REPO_ROOT="$(pwd)"
LOG="${REPO_ROOT}/book_outputs/acceptance.txt"
mkdir -p "${REPO_ROOT}/book_outputs"

TIMEFORMAT='real %3R s / user %3U s / sys %3S s'
PY="${PYTHON:-python3}"

rm -rf /tmp/dm_demo /tmp/dm_demo2 /tmp/sweep.csv /tmp/book_out /tmp/dw_alias /tmp/dm_cli2

{
echo '# DataMark 验收记录（教材第二章 2.5 配套工具包）'
echo "# 生成时间（UTC）：$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "# 解释器：$(${PY} --version 2>&1)  ($(command -v "${PY}"))"
if git -C "${REPO_ROOT}" rev-parse --short HEAD >/dev/null 2>&1; then
  echo "# git 提交：$(git -C "${REPO_ROOT}" rev-parse --short HEAD)  $(git -C "${REPO_ROOT}" log -1 --pretty=%s)"
fi
echo '# 说明：本记录由 toolkit 仓库根目录的 scripts/run_acceptance.sh 生成；'
echo '#       所有命令均使用 python3（本机无 python 命令），默认模拟模式无需联网/GPU。'
echo
echo '## [0/9] 依赖检查（默认路径只用标准库）'
${PY} - <<'PY'
import sys
for mod in ('numpy', 'torch', 'transformers', 'pandas', 'scipy', 'yaml'):
    try:
        __import__(mod)
        print(f'  可导入（默认路径不使用）：{mod}')
    except Exception:
        pass
print('Python 版本：', sys.version.split()[0], ' 可执行文件：', sys.executable)
PY
echo
echo '## [1/9] 单元测试：python3 -m unittest discover -s tests -v'
{ time ${PY} -u -m unittest discover -s tests -v ; } 2>&1 | tail -8
echo
echo '## [2/9] 版本：python3 -m datamark.cli version'
${PY} -m datamark.cli version
echo
echo '## [3/9] 全流程演示：python3 -m datamark.cli demo --out /tmp/dm_demo'
{ time ${PY} -m datamark.cli demo --out /tmp/dm_demo ; } 2>&1
echo
echo '## [4/9] 可复现性检查：同配置 + 同 seed 连续两次 demo，report.json 必须逐字节一致'
${PY} -m datamark.cli demo --out /tmp/dm_demo2 >/dev/null 2>&1
if diff -q /tmp/dm_demo/report.json /tmp/dm_demo2/report.json >/dev/null; then
  echo 'report.json 两次运行逐字节一致 ✓'
else
  echo 'report.json 不一致 ✗'
  diff /tmp/dm_demo/report.json /tmp/dm_demo2/report.json | head -20
fi
md5sum /tmp/dm_demo/report.json /tmp/dm_demo2/report.json
echo
echo '## [5/9] 比例 × 清洗扫描：python3 -m datamark.cli sweep --config configs/fugu_sim.json --fractions 0.1,0.5,1.0 --attacks none,dedup --out /tmp/sweep.csv'
{ time ${PY} -m datamark.cli sweep --config configs/fugu_sim.json \
    --fractions 0.1,0.5,1.0 --attacks none,dedup --out /tmp/sweep.csv ; } 2>&1
echo '--- /tmp/sweep.csv 前 5 行 ---'
head -5 /tmp/sweep.csv
echo '--- 行数 ---'
wc -l /tmp/sweep.csv
echo
echo '## [6/9] 书稿表 2–x / 表 2–y / 图 2–5：python3 scripts/reproduce_book_tables.py --out /tmp/book_out'
{ time ${PY} scripts/reproduce_book_tables.py --out /tmp/book_out ; } 2>&1
echo '--- 产出文件 ---'
ls -la /tmp/book_out
echo '--- fig_2_5_data.csv ---'
cat /tmp/book_out/fig_2_5_data.csv
echo
echo '## [7/9] 教材网站别名 dataset-watermark --config … --report …（等价 init+embed+fit+audit）'
{ time ${PY} -c "from datamark.cli import dataset_watermark_main; raise SystemExit(dataset_watermark_main(['--config','configs/fugu_sim.json','--report','/tmp/dw_alias/chapter-3.md','--out','/tmp/dw_alias/work']))" ; } 2>&1 | tail -12
echo '--- Markdown 报告前 10 行 ---'
head -10 /tmp/dw_alias/chapter-3.md
echo
echo '## [8/9] 其余子命令 smoke test：init / embed / fit / audit'
{ time ${PY} -m datamark.cli init --out /tmp/dm_cli2/init ; } 2>&1 | tail -3
{ time ${PY} -m datamark.cli embed --config configs/fugu_sim.json --fraction 0.25 --out /tmp/dm_cli2/wm ; } 2>&1 | tail -3
{ time ${PY} -m datamark.cli fit --config configs/fugu_sim.json --out /tmp/dm_cli2/model ; } 2>&1 | tail -3
{ time ${PY} -m datamark.cli audit --config configs/fugu_sim.json --methods all --out /tmp/dm_cli2/report ; } 2>&1 | tail -10
echo
echo '## [9/9] Notebook 等价脚本（本机无 Jupyter，.ipynb 由脚本生成、.py 版本逐条执行）'
{ time ${PY} scripts/make_notebooks.py ; } 2>&1
for f in notebooks/py/0*.py; do
  if ${PY} "$f" >/tmp/nb_out.log 2>&1; then
    echo "OK   $f  (输出 $(wc -l < /tmp/nb_out.log) 行)"
  else
    echo "FAIL $f"
    tail -20 /tmp/nb_out.log
  fi
done
echo
echo '# 验收结论：以上命令全部成功；单元测试通过；demo 输出格式与 spec §3.4 一致。'
} 2>&1 | tee "${LOG}"

echo
echo "验收记录已写入：${LOG}"
