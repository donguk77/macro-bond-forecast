"""
22_final_verify_option3.py — 옵션 3 작업물 최종 종합 검증

verification-before-completion: 모든 주장 fresh 명령으로 확인.
한 건이라도 FAIL 시 exit 1.

검증 5 영역:
  A. 옵션 3 정정 항목 8개 모두 파일에 적용됐는지 (grep 검증)
  B. v2 핵심 수치 일관성 (csv 직접 읽고 비교)
  C. 검증·반증 산출물 보존 확인 (verify·package_a·covid)
  D. 모든 산출물 파일 존재 확인 (csv/md/png/script)
  E. 누수 audit fresh 재실행 (CL-05c·CL-04 등 모두 통과)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_V2 = PROJECT_ROOT / 'reports' / 'no_leak_v2'
DOCS_DIR = PROJECT_ROOT / 'docs'

print('=' * 72)
print('22_final_verify_option3.py — 옵션 3 최종 종합 검증')
print('=' * 72)

results = []

def add(check, ok, evidence):
    status = '✅ PASS' if ok else '❌ FAIL'
    results.append((check, status, evidence))
    print(f'\n[{status}] {check}')
    print(f'  → {evidence}')


def file_contains(path, *needles):
    """파일에 모든 needle 이 포함되는지"""
    if not path.exists():
        return False, 'FILE NOT FOUND'
    text = path.read_text(encoding='utf-8')
    missing = [n for n in needles if n not in text]
    if missing:
        return False, f'missing: {missing}'
    return True, f'all {len(needles)} needles found'


# ─────────────────────────────────────────────────────────────────────────
# A. 옵션 3 정정 항목 적용 여부 (grep 검증)
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('A. 옵션 3 정정 항목 적용 검증 (grep)')
print('=' * 72)

# A.1 presentation_v2_revisions.md
ok, ev = file_contains(
    REPORT_V2 / 'presentation_v2_revisions.md',
    '## 4.1', '검증·반증 history',
    '코로나기 hit rate 82.9%는 통계적 우위 입증 어려움',
    '데이터로 반증된 개선안은 채택하지 않음',
    'GARCH', 'regime-switching',
    'Q. "fold1 코로나에서 모델이 잘 대응한 건가? OOD 강건성인가?"',
    'Q. "ACI 같은 추가 개선은 시도해봤나?"',
    'binomial p-value=1.0',
)
add('A.1 presentation_v2_revisions.md 정정 적용', ok, ev)

# A.2 domain_knowledge.md Q19a, Q19b 추가
ok, ev = file_contains(
    DOCS_DIR / 'domain_knowledge.md',
    '### Q19a. 코로나 같은 OOD 충격에 모델이 강건한가?',
    '### Q19b. ACI 같은 conformal prediction 개선은 효과 없나?',
    'binomial p=1.0',
    'over-correction',
)
add('A.2 domain_knowledge.md Q19a/Q19b 추가', ok, ev)

# A.3 comparison_v0_v1_v2.md §8a 신설
ok, ev = file_contains(
    REPORT_V2 / 'comparison_v0_v1_v2.md',
    '## 8a. v2 → v2+ 시도와 반증',
    '데이터로 반증된 개선안은 채택 X',
    '21_verify_covid_robustness.py',
    '20_verify_package_a_direction.py',
    '19_verify_v2.py',
)
add('A.3 comparison_v0_v1_v2.md §8a 신설 + 산출물 갱신', ok, ev)

# A.4 FINAL_option3_summary.md 신설
ok, ev = file_contains(
    REPORT_V2 / 'FINAL_option3_summary.md',
    '옵션 3 최종',
    '시도한 추가 개선 (패키지 A) → 모두 데이터로 반증',
    '한 줄 결론',
    'verification-before-completion',
)
add('A.4 FINAL_option3_summary.md 신설', ok, ev)

# ─────────────────────────────────────────────────────────────────────────
# B. v2 핵심 수치 일관성 (csv 직접)
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('B. v2 핵심 수치 일관성')
print('=' * 72)

# B.1 single-split XGB CQR test dir_acc = 0.6113
xgb_eval = pd.read_csv(REPORT_V2 / 'xgb_v2_eval.csv')
xgb_test_cqr = xgb_eval[(xgb_eval['split'] == 'test') & (xgb_eval['stage'] == 'CQR')]
single_dir = float(xgb_test_cqr['dir_acc_q50'].iloc[0])
add('B.1 single-split dir_acc = 0.6113', abs(single_dir - 0.6113) < 1e-3,
    f'csv = {single_dir:.4f}')

# B.2 walkforward 3-fold 평균 0.6099
wf_xgb = pd.read_csv(REPORT_V2 / 'walkforward_xgb_v2.csv')
wf_cqr = wf_xgb[wf_xgb['stage'] == 'CQR']
mean_dir = float(wf_cqr['dir_acc_q50'].mean())
add('B.2 3-fold 평균 dir_acc ≈ 0.6099', abs(mean_dir - 0.6099) < 1e-3,
    f'재계산 = {mean_dir:.4f}')

# B.3 DM Pooled = -8.78, p < 0.0167
dm = pd.read_csv(REPORT_V2 / 'walkforward_dm_v2.csv')
pooled = dm[dm['fold'] == 'POOLED'].iloc[0]
pooled_dm = float(pooled['DM_HLN'])
pooled_p = float(pooled['p_value'])
add('B.3 Pooled DM=-8.78 + p<0.0167',
    abs(pooled_dm - (-8.782)) < 0.01 and pooled_p < 0.0167,
    f'DM={pooled_dm:.3f}, p={pooled_p:.4f}')

# B.4 leakage audit v2 ≥9 PASS / 0 FAIL
audit = pd.read_csv(REPORT_V2 / 'leakage_audit_v2.csv')
n_pass = (audit['상태'] == '✅').sum()
n_fail = (audit['상태'] == '❌').sum()
add('B.4 leakage_audit_v2 0 FAIL', n_fail == 0,
    f'PASS={n_pass}, FAIL={n_fail}')

# B.5 verification_v2.csv 13/13
ver = pd.read_csv(REPORT_V2 / 'verification_v2.csv')
n_v_pass = ver['status'].str.startswith('✅').sum()
n_v_fail = ver['status'].str.startswith('❌').sum()
add('B.5 verification_v2 13/13 PASS', n_v_fail == 0 and n_v_pass == 13,
    f'PASS={n_v_pass}, FAIL={n_v_fail}, total={len(ver)}')

# ─────────────────────────────────────────────────────────────────────────
# C. 검증·반증 산출물 보존
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('C. 검증·반증 산출물 보존 확인')
print('=' * 72)

# C.1 package_a_verification.md 존재 + 핵심 결론 포함
# (검증 1·4 섹션, ACI, Coverage 모두 있어야)
ok, ev = file_contains(
    REPORT_V2 / 'package_a_verification.md',
    '검증 1',
    '검증 4',
    'ACI',
    'Coverage',
    'rolling vol',  # 부분 매치
)
add('C.1 package_a_verification.md 보존', ok, ev)

# C.2 covid_robustness_check.md 존재
ok, ev = file_contains(
    REPORT_V2 / 'covid_robustness_check.md',
    '코로나',
)
add('C.2 covid_robustness_check.md 보존', ok, ev)

# ─────────────────────────────────────────────────────────────────────────
# D. 모든 산출물 파일 존재 확인
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('D. 모든 산출물 파일 존재 확인')
print('=' * 72)

required_files = {
    # csv (8개)
    'csv/leakage_audit_v2.csv': REPORT_V2 / 'leakage_audit_v2.csv',
    'csv/xgb_grid_v2.csv': REPORT_V2 / 'xgb_grid_v2.csv',
    'csv/xgb_v2_eval.csv': REPORT_V2 / 'xgb_v2_eval.csv',
    'csv/dm_test_xgb_v2.csv': REPORT_V2 / 'dm_test_xgb_v2.csv',
    'csv/walkforward_xgb_v2.csv': REPORT_V2 / 'walkforward_xgb_v2.csv',
    'csv/walkforward_lstm_v2.csv': REPORT_V2 / 'walkforward_lstm_v2.csv',
    'csv/walkforward_dm_v2.csv': REPORT_V2 / 'walkforward_dm_v2.csv',
    'csv/verification_v2.csv': REPORT_V2 / 'verification_v2.csv',
    # md (7개)
    'md/walkforward_summary_v2.md': REPORT_V2 / 'walkforward_summary_v2.md',
    'md/comparison_v0_v1_v2.md': REPORT_V2 / 'comparison_v0_v1_v2.md',
    'md/presentation_v2_revisions.md': REPORT_V2 / 'presentation_v2_revisions.md',
    'md/FINAL_option3_summary.md': REPORT_V2 / 'FINAL_option3_summary.md',
    'md/package_a_verification.md': REPORT_V2 / 'package_a_verification.md',
    'md/covid_robustness_check.md': REPORT_V2 / 'covid_robustness_check.md',
    'md/features_v2_justification.md': DOCS_DIR / 'features_v2_justification.md',
    # script (8개)
    'script/14_features_v2.py': PROJECT_ROOT / 'scripts' / '14_features_v2.py',
    'script/15_xgb_grid_cqr.py': PROJECT_ROOT / 'scripts' / '15_xgb_grid_cqr.py',
    'script/16_walkforward.py': PROJECT_ROOT / 'scripts' / '16_walkforward.py',
    'script/16b_walkforward_save.py': PROJECT_ROOT / 'scripts' / '16b_walkforward_save.py',
    'script/17_full_audit_v2.py': PROJECT_ROOT / 'scripts' / '17_full_audit_v2.py',
    'script/18_build_notebook_08.py': PROJECT_ROOT / 'scripts' / '18_build_notebook_08.py',
    'script/19_verify_v2.py': PROJECT_ROOT / 'scripts' / '19_verify_v2.py',
    'script/20_verify_package_a_direction.py': PROJECT_ROOT / 'scripts' / '20_verify_package_a_direction.py',
    'script/21_verify_covid_robustness.py': PROJECT_ROOT / 'scripts' / '21_verify_covid_robustness.py',
    'script/22_final_verify_option3.py': PROJECT_ROOT / 'scripts' / '22_final_verify_option3.py',
    # png (5개)
    'png/01_dir_acc_3stages.png': REPORT_V2 / 'figures' / '01_dir_acc_3stages.png',
    'png/02_cqr_effect.png': REPORT_V2 / 'figures' / '02_cqr_effect.png',
    'png/03_walkforward_dir_acc.png': REPORT_V2 / 'figures' / '03_walkforward_dir_acc.png',
    'png/04_dm_test.png': REPORT_V2 / 'figures' / '04_dm_test.png',
    'png/05_summary_4panel.png': REPORT_V2 / 'figures' / '05_summary_4panel.png',
    # notebook
    'notebook/08_v2_no_leak_pipeline.ipynb':
        PROJECT_ROOT / 'notebooks' / '08_v2_no_leak_pipeline.ipynb',
    # 데이터
    'data/features_v2_no_leak.csv':
        PROJECT_ROOT / 'data' / 'processed' / 'features_v2_no_leak.csv',
    # 모델
    'model/xgb_v2_q05.json': PROJECT_ROOT / 'models' / 'xgb_v2_q05.json',
    'model/xgb_v2_q50.json': PROJECT_ROOT / 'models' / 'xgb_v2_q50.json',
    'model/xgb_v2_q95.json': PROJECT_ROOT / 'models' / 'xgb_v2_q95.json',
}

missing = [k for k, p in required_files.items() if not p.exists()]
add('D. 산출물 파일 모두 존재',
    len(missing) == 0,
    f'total={len(required_files)}, missing={len(missing)}: {missing[:5]}' if missing else f'all {len(required_files)} files present')

# ─────────────────────────────────────────────────────────────────────────
# E. 누수 audit fresh 재실행
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('E. 누수 audit fresh 재실행')
print('=' * 72)

import os
env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
proc = subprocess.run(
    [sys.executable, '-X', 'utf8', str(PROJECT_ROOT / 'scripts' / '17_full_audit_v2.py')],
    capture_output=True, text=True, encoding='utf-8', env=env,
)
exit_code = proc.returncode
last = proc.stdout.strip().split('\n')[-3:]
add('E. leakage audit fresh re-run exit 0',
    exit_code == 0,
    f'exit={exit_code}, last lines: {" | ".join(last)}')

# ─────────────────────────────────────────────────────────────────────────
# F. 노트북 08 cells 모두 실행됨 확인
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('F. 노트북 08 셀 실행 상태 확인')
print('=' * 72)

nb_path = PROJECT_ROOT / 'notebooks' / '08_v2_no_leak_pipeline.ipynb'
if nb_path.exists():
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    n_code = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    n_executed = sum(1 for c in nb['cells']
                     if c['cell_type'] == 'code' and c.get('execution_count') is not None)
    add('F. 노트북 08 모든 code cells 실행됨', n_executed == n_code,
        f'code cells={n_code}, executed={n_executed}')
else:
    add('F. 노트북 08 존재', False, 'MISSING')

# ─────────────────────────────────────────────────────────────────────────
# G. 핵심 단일 사실 (수치) 보고
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('G. 핵심 수치 단일 진실 (Single Source of Truth)')
print('=' * 72)
print()
print(f'  - v2 single-split test dir_acc CQR : {single_dir:.4f}')
print(f'  - v2 3-fold 평균 dir_acc CQR        : {mean_dir:.4f} ± {wf_cqr["dir_acc_q50"].std():.4f}')
print(f'  - v2 Pooled DM_HLN                  : {pooled_dm:.3f}')
print(f'  - v2 Pooled p-value                 : {pooled_p:.6f}')
print(f'  - leakage audit v2                  : ✅{n_pass} / ❌{n_fail}')
print(f'  - verification v2 13개 검증         : ✅{n_v_pass} / ❌{n_v_fail}')

# ─────────────────────────────────────────────────────────────────────────
# 종합 결과
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('최종 종합 결과')
print('=' * 72)

n_p = sum(1 for _, s, _ in results if s.startswith('✅'))
n_f = sum(1 for _, s, _ in results if s.startswith('❌'))
print(f'\n  PASS: {n_p} / FAIL: {n_f} / Total: {len(results)}')
print()
for ck, st, _ in results:
    print(f'  {st}  {ck}')

# Save
df_v = pd.DataFrame(results, columns=['check', 'status', 'evidence'])
df_v.to_csv(REPORT_V2 / 'final_verification_option3.csv', index=False)
print(f'\n[save] reports/no_leak_v2/final_verification_option3.csv')

if n_f > 0:
    print(f'\n🔴 {n_f} 건 FAIL — 옵션 3 작업 미완료')
    sys.exit(1)
else:
    print('\n🟢 모든 검증 PASS — 옵션 3 발표·제출 준비 완료')
    sys.exit(0)
