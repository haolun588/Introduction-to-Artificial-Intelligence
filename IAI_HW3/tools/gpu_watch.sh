#!/usr/bin/env bash
# 長時間訓練的異常監控：只在異常持續時輸出一行（ANOMALY / RECOVERED），方便 Monitor 逐行通知。
# 用法：bash tools/gpu_watch.sh <LOG> [DONE_REGEX]
#   DONE_REGEX 預設 "ensemble: done"：log 中最後一行以 "[" 開頭的行符合它 -> 印 TRAINING FINISHED 並結束。
#   DONE_PID（環境變數，選用）：該 PID 已不存在也視為結束。
# 追加式 log：應在訓練啟動前或啟動後皆可開；監控啟動時記下 log 檔大小 START_OFF，
#   DONE／ERROR 判定、評估摘要行計數、樣本數行搜尋都只看 START_OFF 之後新增的內容（舊 run 的內容不會誤判）。
# 速度以「樣本/分鐘」計（與 batch size 無關）：由最後一個 "Epoch E: batch b/N" 與最後一個
# "Train <S> ..." 或 "MLM corpus (...) <S> texts ..." 行推算（新內容優先，其次整檔最後一個）；找不到樣本數則用 SAMPLES（預設 8100）。
# 速度時間軸為「有效時間」：每次偵測到新 batch 行，計算與上一筆 batch 行的實際間隔 dt；
#   若這段期間出現了評估摘要行（微調 "Epoch E: train F1="、MLM "Epoch E: train MLM="），
#   該間隔的有效時間改記為「上一個正常間隔」（沒有則記 0），評估時間不計入速率；否則有效時間 += dt。
#   視窗以有效時間計（至少保留 2 點），速率 = 位置差*60/有效時間差；嚴重變慢時下一行晚到，速率即反映；
#   完全沒有新行則由 STALL_SEC 偵測。
# 所有門檻皆可用同名環境變數覆寫。
LOG=${1:?usage: gpu_watch.sh <LOG> [DONE_REGEX]}
DONE_REGEX=${2:-ensemble: done}
INT=${INT:-10}
STALL_SEC=${STALL_SEC:-180}
LOW_UTIL=${LOW_UTIL:-10};   LOW_UTIL_SEC=${LOW_UTIL_SEC:-120}
VRAM_MIB=${VRAM_MIB:-3990}; VRAM_SEC=${VRAM_SEC:-60}
TEMP_C=${TEMP_C:-87};       TEMP_SEC=${TEMP_SEC:-180}
CLK_MHZ=${CLK_MHZ:-1200};   CLK_SEC=${CLK_SEC:-180}
WIN_SEC=${WIN_SEC:-120}      # 有效時間視窗；完整視窗 = 跨度 >= 0.8*WIN_SEC；基準取樣間隔 = WIN_SEC/2
SLOW_RATIO=${SLOW_RATIO:-0.5}; RECOVER_RATIO=${RECOVER_RATIO:-0.6}
MIN_SPM=${MIN_SPM:-500}
SAMPLES=${SAMPLES:-8100}
AC_EVERY=${AC_EVERY:-60}
EVAL_REGEX='Epoch [0-9.]+: train (F1|MLM)='

low=0; vram=0; hot=0; clk=0; stall=0; tick=0
flag_low=0; flag_vram=0; flag_hot=0; flag_clk=0; flag_stall=0; flag_slow=0; flag_bat=0
START_OFF=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
lastsize=$START_OFF
wt=(); wp=()                 # 視窗：有效時間與位置（樣本數）
base_samples=(); base=""; last_base_eff=-1000000
cur_n=""; cur_s=""; last_ep=""; last_b=""; last_p=-1
eff=0; last_real=""; last_dt=0; last_evalcnt=0
u=0; m=0; t=0; c=0; rate=""

ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>=b)}'; }
lt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a<b)}'; }
mul() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%d", a*b}'; }
isnum() { case "$1" in ''|*[!0-9]*) return 1;; *) return 0;; esac; }
newlog() {
  local sz; sz=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  [ "$sz" -lt "$START_OFF" ] && START_OFF=0        # log 被截斷／重建
  tail -c +$((START_OFF+1)) "$LOG" 2>/dev/null
}
reset_stage() { wt=(); wp=(); base_samples=(); base=""; last_base_eff=-1000000; flag_slow=0; last_p=-1; }

pid_alive() {
  kill -0 "$1" 2>/dev/null && return 0
  tasklist //FI "PID eq $1" //NH 2>/dev/null | grep -qw "$1"
}

while true; do
  now=$(date +%s)
  nl=$(newlog)
  # ---- 結束與錯誤判定（只看監控啟動後新增的內容）----
  if printf '%s\n' "$nl" | grep -a "^\[" | tail -1 | grep -aqE "$DONE_REGEX"; then
    echo "TRAINING FINISHED: matched /$DONE_REGEX/"; exit 0
  fi
  if [ -n "$DONE_PID" ] && ! pid_alive "$DONE_PID"; then
    echo "TRAINING FINISHED: process $DONE_PID exited"; exit 0
  fi
  if printf '%s\n' "$nl" | grep -aqE "Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError"; then
    echo "ERROR in training log: $(printf '%s\n' "$nl" | grep -aE 'Traceback|out of memory|OutOfMemoryError|RuntimeError' | tail -1 | cut -c1-200)"; exit 0
  fi

  # ---- GPU 指標（非數字，例如 [N/A]，略過該項比較）----
  q=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  if [ -z "$q" ]; then echo "nvidia-smi query failed"; sleep "$INT"; continue; fi
  IFS=, read -r u m t c <<<"$q"
  if isnum "$u"; then
    [ "$u" -lt "$LOW_UTIL" ] && low=$((low+INT)) || { [ $flag_low = 1 ] && echo "RECOVERED: GPU util back to ${u}%"; low=0; flag_low=0; }
  fi
  if isnum "$m"; then
    [ "$m" -ge "$VRAM_MIB" ] && vram=$((vram+INT)) || { [ $flag_vram = 1 ] && echo "RECOVERED: VRAM ${m} MiB"; vram=0; flag_vram=0; }
  fi
  if isnum "$t"; then
    [ "$t" -ge "$TEMP_C" ] && hot=$((hot+INT)) || { [ $flag_hot = 1 ] && echo "RECOVERED: temp ${t}C"; hot=0; flag_hot=0; }
  fi
  if isnum "$c"; then
    [ "$c" -lt "$CLK_MHZ" ] && clk=$((clk+INT)) || { [ $flag_clk = 1 ] && echo "RECOVERED: SM clock ${c} MHz"; clk=0; flag_clk=0; }
  fi
  size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  [ "$size" = "$lastsize" ] && stall=$((stall+INT)) || { [ $flag_stall = 1 ] && echo "RECOVERED: log growing again"; stall=0; flag_stall=0; }
  lastsize=$size
  [ $low -ge "$LOW_UTIL_SEC" ] && [ $flag_low = 0 ] && { echo "ANOMALY: GPU util <${LOW_UTIL}% for ${LOW_UTIL_SEC}s (now ${u}%, VRAM ${m} MiB)"; flag_low=1; }
  [ $vram -ge "$VRAM_SEC" ] && [ $flag_vram = 0 ] && { echo "ANOMALY: VRAM >=${VRAM_MIB} MiB for ${VRAM_SEC}s (now ${m} MiB, util ${u}%)"; flag_vram=1; }
  [ $hot -ge "$TEMP_SEC" ] && [ $flag_hot = 0 ] && { echo "ANOMALY: GPU temp >=${TEMP_C}C for ${TEMP_SEC}s (now ${t}C, clock ${c} MHz)"; flag_hot=1; }
  [ $clk -ge "$CLK_SEC" ] && [ $flag_clk = 0 ] && { echo "ANOMALY: SM clock <${CLK_MHZ} MHz for ${CLK_SEC}s (now ${c} MHz, temp ${t}C)"; flag_clk=1; }
  [ $stall -ge "$STALL_SEC" ] && [ $flag_stall = 0 ] && { echo "ANOMALY: training log not updated for ${STALL_SEC}s (util ${u}%)"; flag_stall=1; }

  # ---- 訓練速度（樣本/分鐘，有效時間軸）----
  line=$(printf '%s' "$nl" | tail -c 600 | tr '\r' '\n' | grep -aoE "Epoch [0-9]+: batch [0-9]+/[0-9]+" | tail -1)
  if [ -n "$line" ]; then
    read -r ep b n <<<"$(echo "$line" | sed -E 's/Epoch ([0-9]+): batch ([0-9]+)\/([0-9]+)/\1 \2 \3/')"
    s=$(printf '%s\n' "$nl" | grep -aoE "(Train|MLM corpus \([^)]*\)) [0-9]+" | tail -1 | grep -oE "[0-9]+$")
    [ -z "$s" ] && s=$(grep -aoE "(Train|MLM corpus \([^)]*\)) [0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
    [ -z "$s" ] && s=$SAMPLES
    if [ "$ep" != "$last_ep" ] || [ "$b" != "$last_b" ] || [ "$n" != "$cur_n" ] || [ "$s" != "$cur_s" ]; then
      p=$(( (ep-1)*s + b*s/n ))
      if { [ -n "$cur_n" ] && { [ "$n" != "$cur_n" ] || [ "$s" != "$cur_s" ]; }; } || [ "$p" -lt "$last_p" ]; then
        reset_stage                                   # 新 stage（位置倒退或 N/S 改變）
      fi
      cur_n=$n; cur_s=$s; last_ep=$ep; last_b=$b; last_p=$p
      # 有效時間：評估期間的間隔改記上一個正常間隔
      evalcnt=$(printf '%s\n' "$nl" | tr '\r' '\n' | grep -acE "$EVAL_REGEX")
      if [ -n "$last_real" ]; then
        dt=$((now - last_real))
        if [ "$evalcnt" -gt "$last_evalcnt" ]; then dt=$last_dt; else last_dt=$dt; fi
        eff=$((eff + dt))
      fi
      last_real=$now; last_evalcnt=$evalcnt
      wt+=("$eff"); wp+=("$p")
      while [ ${#wt[@]} -gt 2 ] && [ $((eff - wt[0])) -gt "$WIN_SEC" ]; do wt=("${wt[@]:1}"); wp=("${wp[@]:1}"); done
      nw=${#wt[@]}
      if [ "$nw" -ge 2 ]; then
        span=$(( wt[nw-1] - wt[0] ))
        [ -n "$DEBUG" ] && echo "dbg nw=$nw span=$span eff=$eff p=${wp[nw-1]} s=$s n=$n" >&2
        if [ "$span" -gt 0 ] && [ $((span*10)) -ge $((WIN_SEC*8)) ]; then      # 完整視窗
          rate=$(( (wp[nw-1]-wp[0]) * 60 / span ))
          if [ -z "$base" ]; then
            if [ ${#base_samples[@]} -eq 0 ] || [ $((eff - last_base_eff)) -ge $((WIN_SEC/2)) ]; then
              base_samples+=("$rate"); last_base_eff=$eff; [ -n "$DEBUG" ] && echo "dbg sample rate=$rate n=${#base_samples[@]}" >&2
              if [ ${#base_samples[@]} -ge 3 ]; then
                base=$(printf '%s\n' "${base_samples[@]}" | sort -n | sed -n 2p)
                echo "BASELINE: ${base} samples/min (N=${n}, S=${s})"
              fi
            fi
            if [ "$rate" -lt "$MIN_SPM" ]; then
              [ $flag_slow = 0 ] && { echo "ANOMALY: training slowed to ${rate} samples/min over ${WIN_SEC}s (< ${MIN_SPM}; util ${u}%, VRAM ${m} MiB, clock ${c} MHz)"; flag_slow=1; }
            else
              [ $flag_slow = 1 ] && echo "RECOVERED: training speed ${rate} samples/min"; flag_slow=0
            fi
          else
            if lt "$rate" "$(mul "$base" "$SLOW_RATIO")"; then
              [ $flag_slow = 0 ] && { echo "ANOMALY: training slowed to ${rate} samples/min over ${WIN_SEC}s (baseline ${base}; util ${u}%, VRAM ${m} MiB, clock ${c} MHz)"; flag_slow=1; }
            elif ge "$rate" "$(mul "$base" "$RECOVER_RATIO")"; then
              [ $flag_slow = 1 ] && echo "RECOVERED: training speed ${rate} samples/min (baseline ${base})"; flag_slow=0
            fi
          fi
        fi
      fi
    fi
  fi

  # ---- 交流電源（筆電）：BatteryStatus 2 = 使用交流電 ----
  tick=$((tick+1))
  if [ $((tick * INT)) -ge "$AC_EVERY" ] && [ $(( (tick * INT) % AC_EVERY )) -lt "$INT" ]; then
    bs=$(powershell -NoProfile -c "(Get-CimInstance Win32_Battery).BatteryStatus" 2>/dev/null | tr -d '\r ')
    if [ -n "$bs" ] && [ "$bs" != "2" ]; then
      [ $flag_bat = 0 ] && { echo "ANOMALY: laptop not on AC power (BatteryStatus ${bs}); GPU may throttle"; flag_bat=1; }
    elif [ "$bs" = "2" ] && [ $flag_bat = 1 ]; then echo "RECOVERED: back on AC power"; flag_bat=0; fi
  fi
  sleep "$INT"
done
